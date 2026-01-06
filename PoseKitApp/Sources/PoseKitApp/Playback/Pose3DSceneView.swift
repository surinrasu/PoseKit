import RealityKit
import SwiftUI
import simd

struct Pose3DSceneView: UIViewRepresentable {
    var frame: PoseFrame?

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        view.environment.background = .color(.black)

        let anchor = AnchorEntity(world: .zero)
        let renderer = SkeletonRenderer()
        anchor.addChild(renderer.root)

        let camera = PerspectiveCamera()
        anchor.addChild(camera)

        view.scene.anchors.append(anchor)

        context.coordinator.view = view
        context.coordinator.renderer = renderer
        context.coordinator.camera = camera

        let pan = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pan)
        view.addGestureRecognizer(pinch)

        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.update(frame: frame)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator: NSObject {
        var view: ARView?
        var renderer: SkeletonRenderer?
        var camera: PerspectiveCamera?
        let orbit = OrbitController()

        func update(frame: PoseFrame?) {
            guard let frame else { return }
            renderer?.update(frame: frame)
            if let camera {
                camera.transform.matrix = orbit.cameraTransform()
            }
        }

        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let delta = gesture.translation(in: gesture.view)
            orbit.applyPan(delta: delta)
            gesture.setTranslation(.zero, in: gesture.view)
            if let camera {
                camera.transform.matrix = orbit.cameraTransform()
            }
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            orbit.applyPinch(scale: gesture.scale)
            gesture.scale = 1
            if let camera {
                camera.transform.matrix = orbit.cameraTransform()
            }
        }
    }
}

final class SkeletonRenderer {
    let root = Entity()
    private var jointEntities: [ModelEntity] = []
    private var boneEntities: [ModelEntity] = []
    private let edges = SkeletonSchema.edges

    init() {
        let jointMesh = MeshResource.generateSphere(radius: 0.03)
        let jointMaterial = SimpleMaterial(color: .white, isMetallic: false)
        jointEntities = SkeletonSchema.jointOrder.map { _ in
            ModelEntity(mesh: jointMesh, materials: [jointMaterial])
        }
        for joint in jointEntities {
            root.addChild(joint)
        }

        let boneMesh = MeshResource.generateBox(width: 0.02, height: 1, depth: 0.02)
        let boneMaterial = SimpleMaterial(color: .cyan, isMetallic: false)
        boneEntities = edges.map { _ in ModelEntity(mesh: boneMesh, materials: [boneMaterial]) }
        for bone in boneEntities {
            root.addChild(bone)
        }
    }

    func update(frame: PoseFrame) {
        let joints = resolvedJoints(from: frame)
        for (index, entity) in jointEntities.enumerated() where index < joints.count {
            let p = joints[index]
            if p.x.isNaN || p.y.isNaN || p.z.isNaN {
                entity.isEnabled = false
            } else {
                entity.isEnabled = true
                entity.position = SIMD3<Float>(p.x, p.y, p.z)
            }
        }

        for (index, edge) in edges.enumerated() {
            guard let fromIndex = SkeletonSchema.jointOrder.firstIndex(of: edge.0),
                  let toIndex = SkeletonSchema.jointOrder.firstIndex(of: edge.1) else { continue }
            let from = joints[fromIndex]
            let to = joints[toIndex]
            if from.x.isNaN || from.y.isNaN || from.z.isNaN || to.x.isNaN || to.y.isNaN || to.z.isNaN {
                boneEntities[index].isEnabled = false
                continue
            }
            let direction = to - from
            let length = max(0.001, simd_length(direction))
            let midpoint = (from + to) * 0.5

            let bone = boneEntities[index]
            bone.isEnabled = true
            bone.position = midpoint
            bone.scale = SIMD3<Float>(1, length, 1)
            if length > 0.0001 {
                let rotation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: simd_normalize(direction))
                bone.orientation = rotation
            }
        }
    }

    private func resolvedJoints(from frame: PoseFrame) -> [SIMD3<Float>] {
        if hasValidPoints(frame.playback3D) {
            return frame.playback3D
        }
        if hasValidPoints(frame.camera3D) {
            return frame.camera3D
        }
        return planarJoints(from: frame.joints2D)
    }

    private func hasValidPoints(_ points: [SIMD3<Float>]) -> Bool {
        for p in points {
            if !(p.x.isNaN || p.y.isNaN || p.z.isNaN) {
                return true
            }
        }
        return false
    }

    private func planarJoints(from joints2D: [Keypoint2D]) -> [SIMD3<Float>] {
        let jointCount = SkeletonSchema.jointOrder.count
        var joints = joints2D
        if joints.count < jointCount {
            joints += Array(repeating: Keypoint2D(xNorm: 0.5, yNorm: 0.5, score: 0), count: jointCount - joints.count)
        } else if joints.count > jointCount {
            joints = Array(joints.prefix(jointCount))
        }

        let pelvisIndex = SkeletonSchema.jointOrder.firstIndex(of: .pelvisCenter) ?? 0
        let pelvis = pelvisIndex < joints.count ? joints[pelvisIndex] : joints[0]
        let center = SIMD2<Float>(pelvis.xNorm, pelvis.yNorm)

        var minY: Float = 1
        var maxY: Float = 0
        var hasBounds = false
        for joint in joints where joint.score > 0 {
            minY = min(minY, joint.yNorm)
            maxY = max(maxY, joint.yNorm)
            hasBounds = true
        }

        let height = hasBounds ? max(0.001, maxY - minY) : 1
        let scale: Float = 1.6 / height

        return joints.map { joint in
            guard joint.score > 0 else { return SIMD3<Float>(repeating: .nan) }
            let x = (joint.xNorm - center.x) * scale
            let y = (center.y - joint.yNorm) * scale
            return SIMD3<Float>(x, y, 0)
        }
    }
}

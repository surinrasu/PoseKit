import Foundation

/// Joint order and edge list for drawing the skeleton.
public struct SkeletonSchema {
    public static let jointOrder: [Joint] = Joint.storageOrder

    public static let edges: [(Joint, Joint)] = [
        (.leftAnkle, .leftKnee),
        (.leftKnee, .leftHip),
        (.leftHip, .pelvisCenter),
        (.pelvisCenter, .rightHip),
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
        (.leftWrist, .leftElbow),
        (.leftElbow, .leftShoulder),
        (.leftShoulder, .neckCenter),
        (.neckCenter, .rightShoulder),
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        (.neckCenter, .nose),
        (.nose, .leftEye),
        (.nose, .rightEye),
        (.leftEye, .leftEar),
        (.rightEye, .rightEar),
        (.leftShoulder, .rightShoulder),
        (.leftHip, .rightHip)
    ]
}

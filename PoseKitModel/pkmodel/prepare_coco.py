import argparse
import os
import json
import pickle
import cv2
import numpy as np

def main(img_dir, labels_path, output_name, output_img_dir, expand_ratio=1.0, show_points=False):

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)


    with open(labels_path, 'r') as f:
        data = json.load(f)

    img_id_to_name = {}
    img_name_to_id = {}
    for item in data['images']:
        idx = item['id']
        name = item['file_name']
        img_id_to_name[idx] = name
        img_name_to_id[name] = idx
    print(len(img_id_to_name))


    anno_by_imgname = {}
    for annotation in data['annotations']:
        name = img_id_to_name[annotation['image_id']]
        if name in anno_by_imgname:
            anno_by_imgname[name] += [annotation]
        else:
            anno_by_imgname[name] = [annotation]
    print(len(anno_by_imgname))



    new_label = []
    for k,v in anno_by_imgname.items():
        if len(v)>3:
            continue

        img = cv2.imread(os.path.join(img_dir, k))
        h,w = img.shape[:2]
        for idx,item in enumerate(v):
            if item['iscrowd'] != 0:
                continue

            bbox = [int(x) for x in item['bbox']]

            keypoints = item['keypoints']


            keypoints = np.array(keypoints).reshape((17,3))

            keypoints_v = keypoints[keypoints[:,2]>0]
            if len(keypoints_v)<8:
                continue
            min_key_x = np.min(keypoints_v[:,0])
            max_key_x = np.max(keypoints_v[:,0])
            min_key_y = np.min(keypoints_v[:,1])
            max_key_y = np.max(keypoints_v[:,1])

            x0 = min(bbox[0], min_key_x)
            x1 = max(bbox[0]+bbox[2], max_key_x)
            y0 = min(bbox[1], min_key_y)
            y1 = max(bbox[1]+bbox[3], max_key_y)

            cx = (x0+x1)/2
            cy = (y0+y1)/2

            half_size = ((x1-x0)+(y1-y0))/2 * expand_ratio
            new_x0 = int(cx - half_size)
            new_x1 = int(cx + half_size)
            new_y0 = int(cy - half_size)
            new_y1 = int(cy + half_size)

            pad_top = 0
            pad_left = 0
            pad_right = 0
            pad_bottom = 0
            if new_x0 < 0:
                pad_left = -new_x0+1
            if new_y0 < 0:
                pad_top = -new_y0+1
            if new_x1 > w:
                pad_right = new_x1-w+1
            if new_y1 > h:
                pad_bottom = new_y1-h+1

            pad_img = np.zeros((h+pad_top+pad_bottom, w+pad_left+pad_right, 3))
            pad_img[pad_top:pad_top+h,pad_left:pad_left+w] = img
            new_x0 += pad_left
            new_y0 += pad_top
            new_x1 += pad_left
            new_y1 += pad_top

            save_name = k[:-4]+"_"+str(idx)+".jpg"
            new_w = new_x1-new_x0
            new_h = new_y1-new_y0
            save_img = pad_img[new_y0:new_y1,new_x0:new_x1]
            save_bbox = [(bbox[0]+pad_left-new_x0)/new_w,
                         (bbox[1]+pad_top-new_y0)/new_h,
                         (bbox[0]+bbox[2]+pad_left-new_x0)/new_w,
                         (bbox[1]+bbox[3]+pad_top-new_y0)/new_h
                        ]
            save_center = [(save_bbox[0]+save_bbox[2])/2,(save_bbox[1]+save_bbox[3])/2]

            save_keypoints = []
            for kid in range(len(keypoints)):
                save_keypoints.extend([(int(keypoints[kid][0])+pad_left-new_x0)/new_w,
                                       (int(keypoints[kid][1])+pad_top-new_y0)/new_h,
                                       int(keypoints[kid][2])
                                      ])
            other_centers = []
            other_keypoints = [[] for _ in range(17)]
            for idx2,item2 in enumerate(v):
                if item2['iscrowd'] != 0 or idx2==idx:
                    continue
                bbox2 = [int(x) for x in item2['bbox']]#x,y,w,h

                save_bbox2 = [(bbox2[0]+pad_left-new_x0)/new_w,
                             (bbox2[1]+pad_top-new_y0)/new_h,
                             (bbox2[0]+bbox2[2]+pad_left-new_x0)/new_w,
                             (bbox2[1]+bbox2[3]+pad_top-new_y0)/new_h
                            ]
                save_center2 = [(save_bbox2[0]+save_bbox2[2])/2,
                                (save_bbox2[1]+save_bbox2[3])/2]
                if save_center2[0]>0 and save_center2[0]<1 and save_center2[1]>0 and save_center2[1]<1:
                    other_centers.append(save_center2)

                keypoints2 = item2['keypoints']
                keypoints2 = np.array(keypoints2).reshape((17,3))
                for kid2 in range(17):
                    if keypoints2[kid2][2]==0:
                        continue
                    kx = (keypoints2[kid2][0]+pad_left-new_x0)/new_w
                    ky = (keypoints2[kid2][1]+pad_top-new_y0)/new_h
                    if kx>0 and kx<1 and ky>0 and ky<1:
                        other_keypoints[kid2].append([kx,ky])

            save_item = {
                         "img_name":save_name,
                         "keypoints":save_keypoints,
                         "center":save_center,
                         "bbox":save_bbox,
                         "other_centers":other_centers,
                         "other_keypoints":other_keypoints,
                        }

            new_label.append(save_item)



            if show_points:
                cv2.circle(save_img, (int(save_center[0]*new_w), int(save_center[1]*new_h)), 4, (0,255,0), 3)
                for show_kid in range(len(save_keypoints)//3):
                    v = save_keypoints[show_kid*3+2]
                    if v == 1:
                        color = (255,0,0)
                    elif v == 2:
                        color = (0,0,255)
                    elif v > 0:
                        intensity = int(255 * min(max(float(v), 0.1), 1.0))
                        color = (0, 0, intensity)
                    else:
                        continue
                    cv2.circle(save_img, (int(save_keypoints[show_kid*3]*new_w),
                                int(save_keypoints[show_kid*3+1]*new_h)), 3, color, 2)
                cv2.rectangle(save_img, (int(save_bbox[0]*new_w), int(save_bbox[1]*new_h)),
                        (int(save_bbox[2]*new_w), int(save_bbox[3]*new_h)), (0,255,0), 2)
                for show_c in other_centers:
                    cv2.circle(save_img, (int(show_c[0]*new_w), int(show_c[1]*new_h)), 4, (0,255,255), 3)
                for show_ks in other_keypoints:
                    for show_k in show_ks:
                        cv2.circle(save_img, (int(show_k[0]*new_w), int(show_k[1]*new_h)), 3, (255,255,0), 2)


            cv2.imwrite(os.path.join(output_img_dir, save_name), save_img)



    with open(output_name,'w') as f:
        json.dump(new_label, f, ensure_ascii=False)
    print('Total write ', len(new_label))

def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare COCO labels for PoseKitModel")

    parser.add_argument("--img-dir", default=None, help="Image directory for a single run.")
    parser.add_argument("--labels-path", default=None, help="COCO annotations .json for a single run.")
    parser.add_argument("--output-name", default=None, help="Output labels .json for a single run.")

    parser.add_argument("--val-img-dir", default="../../coco2017/val2017")
    parser.add_argument("--val-labels-path", default="../../coco2017/annotations/person_keypoints_val2017.json")
    parser.add_argument("--val-output-name", default="../../dataset/val.json")

    parser.add_argument("--train-img-dir", default="../../coco2017/train2017")
    parser.add_argument("--train-labels-path", default="../../coco2017/annotations/person_keypoints_train2017.json")
    parser.add_argument("--train-output-name", default="../../dataset/train.json")

    parser.add_argument("--output-img-dir", default="../../dataset/imgs")
    parser.add_argument("--expand-ratio", type=float, default=1.0)
    parser.add_argument("--show-points", action="store_true", help="Visualize points on saved images.")
    parser.add_argument("--skip-val", action="store_true", help="Skip the validation split run.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training split run.")

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.img_dir or args.labels_path or args.output_name:
        if not (args.img_dir and args.labels_path and args.output_name):
            raise SystemExit("Must provide --img-dir, --labels-path, and --output-name together.")
        main(
            args.img_dir,
            args.labels_path,
            args.output_name,
            args.output_img_dir,
            expand_ratio=args.expand_ratio,
            show_points=args.show_points,
        )
        raise SystemExit(0)

    if not args.skip_val:
        main(
            args.val_img_dir,
            args.val_labels_path,
            args.val_output_name,
            args.output_img_dir,
            expand_ratio=args.expand_ratio,
            show_points=args.show_points,
        )

    if not args.skip_train:
        main(
            args.train_img_dir,
            args.train_labels_path,
            args.train_output_name,
            args.output_img_dir,
            expand_ratio=args.expand_ratio,
            show_points=args.show_points,
        )

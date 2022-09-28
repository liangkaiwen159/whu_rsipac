import json
import os

test_json_path = '/home/xcy/dataset/chusai_release/test/test_write.json'


class Write_json():

    def __init__(self, ori_json):
        self.json = json.load(open(ori_json, 'r', encoding='UTF-8'))
        self.box_id = 0
        self.img_name2img_id = {}
        for img_name2id_map in self.json['images']:
            self.img_name2img_id[img_name2id_map['file_name']] = img_name2id_map['id']

    def write_info(self, write_infos, image_name):
        for write_info in write_infos:
            write_data = {}
            category_id = int(write_info[5]) + 1
            box = write_info[0:4]
            bbox = [x, y, width, heigth] = int((box[2] + box[0]) / 2), int(
                (box[3] + box[1]) / 2), int(box[2] - box[0]), int(box[3] - box[1])
            segmentation = []
            image_id = self.img_name2img_id[image_name]
            write_data['id'] = self.box_id
            self.box_id += 1
            write_data['image_id'] = image_id
            write_data['category_id'] = category_id
            write_data['iscrowd'] = 0
            write_data['area'] = width * heigth
            write_data['bbox'] = bbox
            write_data['segmentation'] = segmentation
            self.json['annotations'].append(write_data)

    def write_json_file(self, save_path):
        save_file_path = os.path.join(save_path, 'whu_output.json')
        json.dump(self.json, open(save_file_path, 'w'))


if __name__ == '__main__':
    write_json = Write_json(test_json_path)

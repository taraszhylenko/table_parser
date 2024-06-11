import cv2
import PIL.Image

import numpy  as np
import pandas as pd

class CCAnalyzer:
    def __init__(self, image_file, tile_area_floor):
        self.img            = np.array(PIL.Image.open(image_file))
        self.img_grayscale  = CCAnalyzer.to_grayscale(self.img)
        self.img_binary     = CCAnalyzer.to_binary(self.img_grayscale)
        self.img_binary_inv = 255 - self.img_binary
        self.tile_area_floor = tile_area_floor

        self.set_cc_info()
        self.set_cc_neighbors()
        self.set_table_cc_idx()
        self.set_tiles_ccs_idx()

    @staticmethod
    def show(img):
        PIL.Image.fromarray(img).show()
    
    @staticmethod
    def save(img, path):
        PIL.Image.fromarray(img).save(path)

    @staticmethod
    def to_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_binary(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

    @staticmethod
    def get_all_pixels_by_idx(cc_mask, cc_idx):
        pixel_list = list()
        mask_height, mask_length = cc_mask.shape
        for i in range(mask_height):
            for j in range(mask_length):
                cc_label = cc_mask[i, j]
                if cc_label == cc_idx:
                    pixel_list.append((i, j))
        return pixel_list

    @staticmethod
    def color_pixels(img, pixels):
        new_img = img.copy()
        for px_i, px_j in pixels:
            new_img[px_i, px_j] = (0, 255, 0)
        return new_img

    @staticmethod
    def merge_masks(cc_num1, cc_mask1, cc_num2, cc_mask2):
        if cc_mask1.shape != cc_mask2.shape:
            raise AssertionError
        mask_height, mask_length = cc_mask1.shape
        cc_mask = cc_mask1.copy() - 1
        for i in range(mask_height):
            for j in range(mask_length):
                if cc_mask[i, j] == -1:
                    cc_mask[i, j] = cc_mask2[i, j] + cc_num1 - 2
        return cc_mask

    def set_cc_info(self):
        cc_num1, cc_mask1, cc_stats1, cc_cntrs1 = cv2.connectedComponentsWithStats(self.img_binary,     8)
        cc_num2, cc_mask2, cc_stats2, cc_cntrs2 = cv2.connectedComponentsWithStats(self.img_binary_inv, 8)
        self.cc_num   = cc_num1 + cc_num2 - 1
        self.cc_mask  = CCAnalyzer.merge_masks(cc_num1, cc_mask1,  cc_num2, cc_mask2 )
        self.cc_stats = np.vstack([cc_stats1[1:], cc_stats2[1:]])
        self.cc_cntrs = np.vstack([cc_cntrs1[1:], cc_cntrs2[1:]])

    def set_cc_neighbors(self):
        self.cc_neighbors = dict()
        l, h = self.cc_mask.shape
        for i in range(l - 1):
            for j in range(h - 1):
                cc = self.cc_mask[i, j]
                cc_vert = self.cc_mask[i+1, j  ]
                cc_horz = self.cc_mask[i  , j+1]
                cc_diag = self.cc_mask[i+1, j+1]
                for cc_neighbor in [cc_vert, cc_horz, cc_diag]:
                    if cc != cc_neighbor:
                        if cc not in self.cc_neighbors:
                            self.cc_neighbors[cc] = {cc_neighbor}
                        else:
                            self.cc_neighbors[cc].update([cc_neighbor])
                        if cc_neighbor not in self.cc_neighbors:
                            self.cc_neighbors[cc_neighbor] = {cc}
                        else:
                            self.cc_neighbors[cc_neighbor].update([cc])

    def set_table_cc_idx(self):
        self.cc_tableness = pd.DataFrame(self.cc_stats).assign(tableness = lambda df: df[2] * df[3] / df[4]).tableness
        self.cc_table_idx = self.cc_tableness.sort_values(ascending=False).index[0]
    
    def set_tiles_ccs_idx(self):
        tx, ty, tdx, tdy, _ = self.cc_stats[self.cc_table_idx]
        table_neighbors = self.cc_neighbors[self.cc_table_idx]
        new_img = self.img.copy()
        for nb in table_neighbors:
            x, y, dx, dy, area = self.cc_stats[nb]
            if (tx <= x <= x + dx <= tx + tdx) and (ty <= y <= y + dy <= ty + tdy) and area > self.tile_area_floor:
                new_img = cv2.rectangle(new_img, (x, y), (x + dx, y + dy), (0, 255, 0), 1)
                CCAnalyzer.save(self.img[y:y+dy, x:x+dx], f'/home/steganopus/Documents/TAoS/misc/20240609/table_parser/data/split/{nb}.jpg')
        CCAnalyzer.show(new_img)
        breakpoint()
        print(123)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file',      type=str, required=True)
    parser.add_argument('--tile-area-floor', type=str, required=True)
    args = parser.parse_args()

    cca = CCAnalyzer(args.image_file, int(args.tile_area_floor))
    cca.show(cca.img)
    cca.show(cca.img_grayscale)
    cca.show(cca.img_binary)


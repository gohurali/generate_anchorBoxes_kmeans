import numpy as np
import torch
import torch.utils.data
import yaml
import cv2
import json
import xml
from xml.etree import ElementTree
import os
import sys
import re
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split

config = yaml.safe_load(open("config.yaml"))


class Dataset:

    def __init__(self,config):
        pass

    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

    def open_traffic_ds(self,config):
        imgs = []
        annots = []
        # Get images
        for idx,im_file in (enumerate(sorted(os.listdir(config["ds_loc"]+"images/"),key=self.natural_keys))):
            if(config['data_count'] == 'full'):
                xml_file = im_file.split(".")[0] + ".xml"
                tree = ElementTree.parse(config["ds_loc"]+"annotations/"+xml_file)
                im_data = get_xml_data(tree)

                img = cv2.imread(config["ds_loc"]+"images/"+im_file)
                imgs.append(img)
                annots.append(im_data)
            else:
                if(idx == config['data_count']):
                    break
                xml_file = im_file.split(".")[0] + ".xml"
                tree = ElementTree.parse(config["ds_loc"]+"annotations/"+xml_file)
                im_data = self.get_xml_data(tree)

                img = cv2.imread(config["ds_loc"]+"images/"+im_file)
                imgs.append(img)
                annots.append(im_data)
        
        return imgs,annots

    def get_xml_data(self,tree):
        root = tree.getroot()
        bboxes = []
        file_name = ''
        w = 0
        h = 0
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        im_data = {
            'filename': file_name,
            'w' : 0,
            'h' : 0,
            'bboxes': bboxes
        }
        for item in root:
            if(item.tag == 'filename'):
                im_data['filename'] = item.text
            if(item.tag == "size"):
                im_data['w'] = int(item[0].text)
                im_data['h'] = int(item[1].text)
            if(item.tag == "object"):
                for obj_items in item:
                    if(obj_items.tag == "bndbox"):
                        bbox_data = {
                            'x_min' : 0,
                            'x_max' : 0,
                            'y_min' : 0,
                            'y_max' : 0,
                            'x_mid' : 0,
                            'y_mid' : 0
                        }
                        bbox_data['x_min'] = int(obj_items[0].text)
                        bbox_data['y_min'] = int(obj_items[1].text)
                        bbox_data['x_max'] = int(obj_items[2].text)
                        bbox_data['y_max'] = int(obj_items[3].text)
                        
                        # Calculating the mid-point of the bbox
                        bbox_data['x_mid'] = np.floor( 1/2 * (bbox_data['x_min'] + bbox_data['x_max']) )
                        bbox_data['y_mid'] = np.floor( 1/2 * (bbox_data['y_min'] + bbox_data['y_max']) )
                        bboxes.append(bbox_data)
        im_data['bboxes'] = bboxes
        return im_data

class DataPrepper:
    
    def __init__(self, x_data=None, y_data=None):
        self.x_data = x_data
        self.y_data = y_data
        pass

    def rescale_data(self,x_data,y_data):
        """
        In order for the model to learn the locations of the objects
        we need to rescale the images and bounding boxes to the same
        height and width
        
        Due to the fact that the data is given in varying dimensions,
        we need to set it to a set size
        """
        
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            
            # Resize img
            im_resized = cv2.resize(img,(config['img_size'],config['img_size']))
            x_data[idx] = im_resized
            
            # Rescale y_data accordingly
            x_scaler = config['img_size'] / y['w']
            y_scaler = config['img_size'] / y['h']
            
            y['w'] = config['img_size']
            y['h'] = config['img_size']
            
            # Rescale each bbox
            for bbox in y['bboxes']:
                bbox['x_min'] = int(( bbox['x_min'] * x_scaler ))
                bbox['x_max'] = int(( bbox['x_max'] * x_scaler ))

                bbox['y_min'] = int(( bbox['y_min'] * y_scaler ))
                bbox['y_max'] = int(( bbox['y_max'] * y_scaler ))

                bbox['x_mid'] = int(( bbox['x_mid'] * x_scaler ))
                bbox['y_mid'] = int(( bbox['y_mid'] * y_scaler ))
            
            y_data[idx] = y
        return x_data, y_data
        
    
    def normalize(self,x_data,y_data):
        for idx,img in enumerate(x_data):
            y = y_data[idx]
            
            # normalize img
            img = img[:, :, ::-1].transpose(0,1,2)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            
            y['w'] = config['img_size']/config['img_size']
            y['h'] = config['img_size']/config['img_size']
            
            # normalize bboxes
            for bbox in y['bboxes']:
                bbox['x_min'] /= config['img_size']
                bbox['x_max'] /= config['img_size']

                bbox['y_min'] /= config['img_size']
                bbox['y_max'] /= config['img_size']

                bbox['x_mid'] /= config['img_size']
                bbox['y_mid'] /= config['img_size']
            
            x_data[idx] = img
            y_data[idx] = y
        return x_data, y_data
    
    def draw_grid(self,img,y_data):
        fig,ax = plt.subplots(1)
        # Visualize the grid cells that YOLO will be using to track an objects location
        x = np.floor(y_data['w'] / config['num_grid_cells'])
        y = np.floor(y_data['h'] / config['num_grid_cells'])

        print(x)
        print(y)

        move_x = x
        move_y = y
        #if(config['jupyter']):
        ax.imshow(img)
        for i in range(config['num_grid_cells']):
            plt.plot([move_x,move_x],[0,y_data['h']],color='r',marker='.')
            plt.plot([0,y_data['w']],[move_y,move_y],color='r',marker='.')
            move_x += x
            move_y += y
        if(config['jupyter']):
            plt.show()
        else:
            plt.savefig(config['save_plots_loc']+'grid_im.png')
            #plt.imsave(config['save_plots_loc']+'grid_im.png',img)
        
    def visualize_data(self,img,data):
        print(data['h'])
        print(data['w'])
        fig,ax = plt.subplots(1)
        if(config['jupyter']):
            ax.imshow(img)
        
        for bbox in data['bboxes']:
            rect = patches.Rectangle(
                (bbox['x_min'],bbox['y_min']),
                bbox['x_max']-bbox['x_min'],
                bbox['y_max']-bbox['y_min'],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            pt = plt.plot(bbox['x_mid'],bbox['y_mid'],color='r',marker='.')
            ax.add_patch(rect)
        if(config['jupyter']):
            plt.show()
        else:
            plt.imsave(config['save_plots_loc']+'visualize.png',img)
        plt.cla()
        plt.clf()
        plt.close()
import random
import numpy as np
import yaml
import cv2
import json
import xml
from xml.etree import ElementTree
import os
import sys
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config = yaml.safe_load(open("config.yaml"))

class KMeans:
    def __init__(self,k,dataset):
        self.num_clusters = k
        self.bboxes = []
        for example in dataset:
            for bbox in example['bboxes']:
                # Placing boxes at the origin
                self.bboxes.append({
                    'x_min' : 0,
                    'y_min' : 0,
                    'x_max' : bbox['x_max'] - bbox['x_min'],
                    'y_max' : bbox['y_max'] - bbox['y_min'],
                    'x_mid' : 1/2 * (bbox['x_max'] - bbox['x_min']),
                    'y_mid' : 1/2 * (bbox['y_max'] - bbox['y_min'])
                })
        pass
    
    def gen_random_centroids(self):
        # First generate the random indicies
        #print(len(self.bboxes))
        rand_idx = []
        rand_idx = [ random.randrange(len(self.bboxes)) for i in range(self.num_clusters) ]
        centroids = [ self.bboxes[i] for i in rand_idx ]
        return centroids
    
    def create_clusters(self):
        # Create lists for the number of clusters
        # each list represents a cluster
        clusters = []
        for i in range(self.num_clusters):
            clusters.append([])        
        return clusters
    
    def visualize_data(self,bboxes,xlim_l=-1,xlim_r=300,ylim_l=-1,ylim_r=300):
        fig,ax = plt.subplots()
        
        for bbox in bboxes:
            rect = patches.Rectangle(
                (bbox['x_min'],bbox['y_min']),
                bbox['x_max']-bbox['x_min'],
                bbox['y_max']-bbox['y_min'],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
        
        plt.xlim(xlim_l,xlim_r)
        plt.ylim(ylim_l,ylim_r)
        
        plt.show()
    
    def iou(self,bbox1,bbox2,visualize=False,debug=False):
        max_min_x = max(bbox1['x_min'],bbox2['x_min'])
        min_max_x = min(bbox1['x_max'],bbox2['x_max'])
        intersection_x = min_max_x - max_min_x
        
        max_min_y = max(bbox1['y_min'],bbox2['y_min'])
        min_max_y = min(bbox1['y_max'],bbox2['y_max'])
        intersection_y = min_max_y - max_min_y
        
        intersection_area = intersection_x * intersection_y
        if(intersection_x <= 0 or intersection_y <= 0 or intersection_area <= 0):
            iou = 0
        else:
            # Calculate Area of each box
            bbox1_area = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
            bbox2_area = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])

            iou = (intersection_area)/(bbox1_area + bbox2_area - intersection_area)
        return iou
        
    def iou_2(self,centroids,bbox,visualize=False,debug=False):
        """
        iou_2 is very similar to iou, the difference is that
        centroids in this case is an array of bboxes which is
        of size (k)
        """
        cluster_iou = []
        for c_box in centroids:
                       
            max_min_x = max(c_box['x_min'],bbox['x_min'])
            min_max_x = min(c_box['x_max'],bbox['x_max'])
            intersection_x = min_max_x - max_min_x 
            
            max_min_y = max(c_box['y_min'],bbox['y_min'])
            min_max_y = min(c_box['y_max'],bbox['y_max'])
            intersection_y = min_max_y - max_min_y
            
            intersection_area = intersection_x * intersection_y
            
            if(debug):
                print('i_x = ', intersection_x)
                print('i_y = ', intersection_y)
                print("I_A = ", intersection_area)
            
            if(intersection_x <= 0 or intersection_y <= 0 or intersection_area <= 0):
                # If boxes are not intersected
                iou = 0
                if(visualize):
                    self.visualize_data(bbox,c_box)
            else:
                # Calculate Area of each box
                bbox_area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                c_box_area = (c_box['x_max'] - c_box['x_min']) * (c_box['y_max'] - c_box['y_min'])
                iou = (intersection_area)/(bbox_area + c_box_area - intersection_area)
            if(debug):
                print("iou = ", iou)
            cluster_iou.append(iou)
        if(debug):
            print(cluster_iou)
        return np.array(cluster_iou)
    
    def avg_iou(self,dataset,centroids): 
        sum = 0
        for bbox in self.bboxes:
            sum += max(self.iou_2(centroids,bbox))
        return sum/len(self.bboxes)
    
    def recalculate_centroids(self,centroids,clusters,debug=False):
        cluster_sums = []
        for cluster in clusters:
            cluster_sum = {
                        'x_min' : 0,
                        'x_max' : 0,
                        'y_min' : 0,
                        'y_max' : 0,
                        'x_mid' : 0,
                        'y_mid' : 0
                    }
            for bbox in cluster:
                cluster_sum['x_min'] += bbox['x_min']
                cluster_sum['y_min'] += bbox['y_min']
                cluster_sum['x_max'] += bbox['x_max']
                cluster_sum['y_max'] += bbox['y_max']
                cluster_sum['x_mid'] += bbox['x_mid']
                cluster_sum['y_mid'] += bbox['y_mid']
            cluster_sums.append(cluster_sum)
        
        # Calculate the mean for each sum
        for i in range(self.num_clusters):
            if(len(clusters[i]) > 0):
                centroids[i]['x_min'] = cluster_sums[i]['x_min']/len(clusters[i])
                centroids[i]['y_min'] = cluster_sums[i]['y_min']/len(clusters[i])
                centroids[i]['x_max'] = cluster_sums[i]['x_max']/len(clusters[i])
                centroids[i]['y_max'] = cluster_sums[i]['y_max']/len(clusters[i])
                centroids[i]['x_mid'] = cluster_sums[i]['x_mid']/len(clusters[i])
                centroids[i]['y_mid'] = cluster_sums[i]['y_mid']/len(clusters[i])
                
        if(debug):
            self.visualize_data(centroids[0],centroids[1])
        return centroids
    
    def write_anchors(self,centroids):
        anchors = {}
        for idx,anchor in enumerate(centroids):
            anchors["anchor"+str(idx+1)] = anchor
        json_anchors = json.dumps(anchors,sort_keys=False, indent=4)
        anchor_file = open("anchors.json","w+")
        
        sys.stdout = anchor_file
        print(json_anchors)
        sys.stdout = sys.__stdout__
        anchor_file.close()
        print("[SUCCESS]: Anchor boxes written to file.")
        
    def fit(self,max_iterations=1,debug=False):
        # generate the init random centroids
        self.centroids = self.gen_random_centroids()
        
        # Create the initial clusters
        clusters = self.create_clusters()

        old_centroids = None
        old_min_clusters = np.ones(len(self.bboxes))*(-1)
        
        iteration = 0
        while True:
            distances = []
            # Compare boxes to centroids
            for bbox in self.bboxes:
                cluster_iou = []
                for centroid in self.centroids:
                    iou = self.iou(centroid,bbox)
                    cluster_iou.append(iou)
                cluster_iou = np.array(cluster_iou)
                iou_distance = 1 - cluster_iou
                distances.append(iou_distance)
            distances = np.array(distances)
            
            min_clusters = np.argmin(distances,axis=1)
            
            if(min_clusters == old_min_clusters).all():
                print("DONE! Iteration = {} ".format(iteration))
                print("Accuracy = ",self.avg_iou(self.bboxes,self.centroids))
                if(debug):
                    print("Anchors: ", self.centroids)
                return self.centroids
            
            #print(len(min_clusters))
            #print(len(self.bboxes))
            
            # Assigning bboxes to their respective nearest clusters
            for idx,bbox in enumerate(self.bboxes):
                cluster_idx = min_clusters[idx]
                clusters[cluster_idx].append(bbox)
            
            # Recalculate the centroids for each cluster
            self.centroids = self.recalculate_centroids(self.centroids,clusters)
            
            # Save the previous min_clusters to check if the clusters change in the next iteration
            old_min_clusters = min_clusters.copy()
            iteration += 1
            
    def fit_average(self,max_iterations=1):
        centroids = self.gen_random_centroids()
        all_anchors = []
        for kmeans_iteration in range(max_iterations):
            self.centroids = self.fit()
            all_anchors.append(self.centroids)
        # Average of all anchors
        self.centroids = self.recalculate_centroids(self.centroids,all_anchors) 
        return self.centroids 

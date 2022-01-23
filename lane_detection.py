import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

'''
Possible improvements:
    - Removing Car and Tire gradients check not across whole image but only center where the car is (cut_gray , line 76) -> Save computation time
    - Find start points in lane detection not on front row of the car but bottom row of the image bzw consider whole image(find_first_lane_point)
    - Start point search uses road width of 20. If car is positioned skewed or orthogonal, road width is higher up to infinity -> no start points detected
    - Instead of looped neighbourhood search across rows, neighbourhood search in local neighbourhood of found points (lane_detection)
    - center line calculation ahead even if only on lane is visible
'''

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=68, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0

        # consider only green values of the street segments, gras parts and the curve markings to calculate the lane borders
        # street_colors = np.array([102,105,107])
        # gras_colors = np.array([204,229,230])         # bright green is sometimes 229 or 230
        # curve_markings_colors = np.array([0,255])
        # self.color_gradients = np.concatenate((np.abs([street_colors- border_colors for border_colors in np.concatenate((gras_colors,curve_markings_colors))])))


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the imagen at the front end of the car (e.g. pixel row 68) 
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1

        '''
        # only Green - Street: (102,102,102) or (105,105,105) or (107,107,107), Gras: (102,229,102) or (102,204,102), Curve Markings: (255,255,255) or (255,0,0), font of the car (last bottom row) (204,0,0), tires (0,0,0)
        # going straight, only the center bottom row show the 204 red front of the car. In some manouviers the front of the car can extend in the second bottom row and tires can be seen. Also motion blure can mix car and tire colors.
        # A difference in more than 97 in the green value is an edge of the road
        h_diff = np.abs(state_image_full[:68,:,1] - np.hstack((state_image_full[:68,1:,1],np.ones((68,1))*204)))
        v_diff = np.abs(state_image_full[:68,:,1] - np.vstack((np.ones((1,96))*204,state_image_full[:67,:,1])))
        total_grad = (h_diff == 102) | (h_diff == 99) | (h_diff == 97) | (h_diff == 127) | (h_diff == 124) | (h_diff == 122) | (h_diff == 128) | (h_diff == 125) | (h_diff == 123) | (h_diff == 105) | (h_diff == 107) | (h_diff == 153) | (h_diff == 150) | (h_diff == 148) | (v_diff == 102) | (v_diff == 99) | (v_diff == 97) | (v_diff == 127) | (v_diff == 124) | (v_diff == 122) | (v_diff == 128) | (v_diff == 125) | (v_diff == 123) | ( v_diff == 105) | ( v_diff == 107) | ( v_diff == 153) | ( v_diff == 150) | (v_diff == 148)

        removed_car = state_image_full[:68,:,0] != 204 # remove car gradients - only in lower half
        removed_tire = np.sum(state_image_full[:68,:,:],axis=2) != 0 # remove tire gradients - only in lower half - comes in (0,0,0) and (76,76,76) = 228
        #removed_tire_brighter = np.sum(state_image_full[:68,:,:],axis=2) != 228 # not needed
        total_grad[67,45] = False # remove left part outside car that has a positive value from the shift
        total_grad = total_grad & removed_car & removed_tire 
        total_grad[0,:] = False # remove boarder from shifting 
        total_grad[:,95] = False # remove boarder from shifting

        return total_grad[::-1] # flip image for some reason


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        '''
        # Implemented in cut_gray


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        # no need for this function

    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 68x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        lane_boundary1_startpoint = []
        lane_boundary2_startpoint = []
        while not lanes_found:
            if row == 68:
                return np.array([]), np.array([]),False
            line_points = np.where(gradient_sum[row] == 1)[0]
            #line_points.astype(int)
            if(line_points.size == 2):
                # assigning smaller point to left and higher value point to right lane 
                lane_boundary1_startpoint.append(line_points.min())
                lane_boundary2_startpoint.append(line_points.max())
                lanes_found = True
            elif(line_points.size > 2):
                # calculate distance between all found points. Choose the point pair, whoes distance is closest to the road width of 20px (when centered)
                # if multiple close candidates exists (road_diff < 6) choose the two starting points, that are closest to the center of the image
                index_arr = np.zeros((np.arange(line_points.size).sum(),2),dtype=int)
                count = 0
                line_points_diff = []
                for i in range(line_points.size):
                    for j in range (i+1,line_points.size):
                        line_points_diff.append(np.abs(line_points[i] - line_points[j]))
                        index_arr[count] = [i,j]
                        count += 1
                line_points_diff = np.array(line_points_diff)
                road_diff = (np.abs(line_points_diff - 20))
                index = np.argsort(road_diff)
                best_index_array = []
                if np.sum(road_diff < 6) > 0:
                    for i in range(np.sum(road_diff < 6)):
                        best_index_array.append(np.abs(index_arr[index[i]].mean() - 48))
                    best_index_array = np.array(best_index_array)
                    best_index = index[best_index_array.argmin()]
                else: best_index = 0
                lane_boundary1_startpoint.append(line_points[index_arr[best_index][0]])
                lane_boundary2_startpoint.append(line_points[index_arr[best_index][1]])
                lanes_found = True
            else:
                # less than 2 lane points found, go to next row
                row += 1
        lane_boundary1_startpoint = np.array(lane_boundary1_startpoint)
        lane_boundary2_startpoint = np.array(lane_boundary2_startpoint)

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray + find gradients
        gradient_sum = self.cut_gray(state_image_full)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        x1_spline = np.array([0])
        x2_spline = np.array([0])

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
            
            
            # Possible better implementations: find connected components with of graph representation (BFS)
            # But speperated Loop only runs max 2 times, mostly only once or not at all on straight curves
            
            row = 1
            seperated_road_part = np.array([[0,0]])
            while row < 68:
                line_points = np.where(gradient_sum[row] == 1)[0]
                for point in line_points:
                    point = int(point)

                    dist1 = np.min((point - lane_boundary1_points)**2 + (row - x1_spline)**2)
                    dist2 = np.min((point - lane_boundary2_points)**2 + (row - x2_spline)**2)

                    # If point is too far away from line, it may belong to a different part of the road, looping back after a turn
                    # To assignt this part to the correct lane, assign the rest first and store all far away points in seperated_road_part.
                    # Loop over seperated parts until all cound be assigned a lane
                    if dist1 > 25 and dist2 > 25:
                        seperated_road_part = np.vstack((seperated_road_part,[[row,point]]))
                        continue
                    if dist1 < dist2:
                        lane_boundary1_points = np.append(lane_boundary1_points,point)
                        x1_spline = np.append(x1_spline,row)
                    else:
                        lane_boundary2_points = np.append(lane_boundary2_points,point)
                        x2_spline = np.append(x2_spline,row)
                row += 1
            # old_size: check if sperated part got smaller in loop, if not break
            old_size = 999999999999
            new_seperated = np.array([[0,0]])
            # Loop over seperated parts
            if seperated_road_part.size > 1:
                sep = True
                while seperated_road_part.size < old_size and seperated_road_part.size > 0:
                    old_size = seperated_road_part.size
                    for row, point in seperated_road_part[::-1][:-1]:
                        dist1 = np.min((point - lane_boundary1_points) ** 2 + (row - x1_spline) ** 2)
                        dist2 = np.min((point - lane_boundary2_points) ** 2 + (row - x2_spline) ** 2)

                        if dist1 > 25 and dist2 > 25:
                            new_seperated = np.vstack((new_seperated,[[row,point]]))
                            continue
                        if dist1 < dist2:
                            lane_boundary1_points = np.append(lane_boundary1_points,point)
                            x1_spline = np.append(x1_spline,row)
                            #lane_1[row,point] = 1
                        else:
                            lane_boundary2_points = np.append(lane_boundary2_points,point)
                            x2_spline = np.append(x2_spline,row)
                            #lane_2[row,point] = 1
                    seperated_road_part = new_seperated

            # Cut down to smallest common lane length by finding nearest point on the longer lane from the end point of the short lane
            # That way a orthogonal cut off is generated, allowing for easy center lane calculation
            if lane_boundary1_points.size < lane_boundary2_points.size - 10 :
                cut_off_index = ((lane_boundary2_points - lane_boundary1_points[-1])**2 + (x2_spline - x1_spline[-1])**2).argmin()
                x2_spline = x2_spline[:cut_off_index]
                lane_boundary2_points = lane_boundary2_points[:cut_off_index]
            
            elif lane_boundary2_points.size < lane_boundary1_points.size - 10 :
                cut_off_index = ((lane_boundary1_points - lane_boundary2_points[-1])**2 + (x1_spline - x2_spline[-1])**2).argmin()
                x1_spline = x1_spline[:cut_off_index]
                lane_boundary1_points = lane_boundary1_points[:cut_off_index]

            
            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:

                # Pay attention: the first lane_boundary point might occur twice
                #lane 1
                tck, u = splprep([x1_spline,lane_boundary1_points], s=self.spline_smoothness)
                lane_boundary1 = tck

                #lane 2
                tck, u = splprep([x2_spline,lane_boundary2_points], s=self.spline_smoothness)
                lane_boundary2 = tck
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2


        # other return for additional visualization
        #return [[lane_boundary1, lane_boundary2],gradient_sum, [lane_1, lane_2]]
        return [lane_boundary1, lane_boundary2]

    def plot_state_lane(self, state_image_full, steps, fig, gradient_sum=0, lane_points=0, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 15)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        # commented out additional visualition
        plt.gcf().clear()
        #plt.subplot(221)
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[1], lane_boundary1_points_points[0]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[1], lane_boundary2_points_points[0]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[1], waypoints[0]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        fig.canvas.flush_events()

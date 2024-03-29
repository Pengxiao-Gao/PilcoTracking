import numpy as np
from numpy.linalg import inv
from pathplanning.libs.extend_transformations import *
from math import fabs
import copy 
from scipy import spatial


class Analysis_Error() :
    def __init__(self):
        pass
    
    def get_lateralError_byKdTree(self, desired_path, driven_path=None):
        # Get index and error-distance by kdtree
        # Get sign by generating the tf-matrixes and the delta-tf-matrix and the sign of matrix[1,3]= sign(y-Position)
        print(">> Analysis_Error - get_lateralError_byKdTree(...) <<")

        # Generate kdTee of
        # print()
        drv_x, drv_y, drv_yaw = driven_path[0], driven_path[1], driven_path[2]
        tree = spatial.KDTree( list(zip(drv_x, drv_y ) ) )

        des_x, des_y, des_yaw = desired_path[0].reshape(-1, 1), desired_path[1].reshape(-1, 1), desired_path[2].reshape(-1, 1)
        des_xy = np.hstack( (des_x, des_y) )
        lat_error, idxs = tree.query(des_xy)
        lat_error = lat_error.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

        # Determine Sign of lateral error
        for idx, val in np.ndenumerate(des_x):
            x, y, yaw = des_x[idx], des_y[idx], des_yaw[idx]
            map2desired = xyYaw_to_matrix(x, y, yaw)

            x, y, yaw = drv_x[ idxs[idx] ], drv_y[ idxs[idx] ], drv_yaw[ idxs[idx] ]
            map2driven = xyYaw_to_matrix(x, y, yaw)

            desired_2_driven = np.linalg.inv(map2desired) @ map2driven
            # print("new dL = ", np.hypot(desired_2_driven[0,3], desired_2_driven[1,3]))
            sign_lateral_error = np.sign(desired_2_driven[1, 3])
            lat_error[idx] *= sign_lateral_error

        # debug
        print("First Desired Pose:", des_x[0], des_y[0], "; drv pose:", drv_x[0], drv_y[0])

        return lat_error


    def print_keyNumbers(self, data, data_for=""):
        data = data[ ~np.isnan(data) ]
        # print(locals().iteritems())
        # my_var_name = [ k for k,v in locals().iteritems() if v == data][0]

        print("KeyNumers for", data_for)
        print("\t Not absolute:")
        print("\t\t min:", data.min() )
        print("\t\t max:", data.max() )
        print("\t\t mean:", np.mean(data) )
        print("\t\t var:", np.var(data) )
        print("\t\t std:", np.std(data) )

        print("\t Absolute:")
        print("\t\t max:", np.fabs(data).max() )
        print("\t\t mean:", np.mean( np.fabs(data) ) )
        print("\t\t var:", np.var( np.fabs(data) ) )
        print("\t\t std:", np.std( np.fabs(data) ) )



    def get_mean(self, data, filter_nan=False):
        if filter_nan == True:
            data = data[ ~np.isnan(data) ]

        return np.mean(np.fabs(data))

    def lateral_error(self, ref_path, driven_path):
        
        idx_range = 10
        min_range = 10
        max_range = 1000
       
        print("ref_path: ", len(ref_path), ref_path[0].shape, ref_path[1].shape, ref_path[2].shape )
        # ref = np.hstack( (ref_path[0], ref_path[1], ref_path[2]) ).ravel()
        ref=np.column_stack( (ref_path[0], ref_path[1], ref_path[2]))


        drv = np.column_stack( (driven_path[0], driven_path[1], driven_path[2]) )

        lat_error = np.ones( shape=(ref.shape[0],1 ) ) * 100000.0


        print(drv)
        print(driven_path[0][0], driven_path[1][0], driven_path[2][0])
        print(driven_path[0][-1], driven_path[1][-1], driven_path[2][-1])

        # return 0
        last_idx = 0

        for idxR in range(ref.shape[0]) :
            refMat = xyYaw_to_matrix(ref[idxR, 0], ref[idxR, 1], ref[idxR, 2] )
            ref2g = inv(refMat)

            start_idx = last_idx - idx_range
            if start_idx < 0: start_idx = 0
            end_idx = last_idx + idx_range
            if end_idx > drv.shape[0]: end_idx = drv.shape[0]

            # print("new search", idxR, last_idx, start_idx, end_idx)

            cL = 1000000.0

            new_last_idx = copy.deepcopy(last_idx)

            for idxD in range(start_idx, end_idx) :
            # for idxD in range( drv.shape[0] ) :
                g2drv = xyYaw_to_matrix(drv[idxD, 0], drv[idxD, 1], drv[idxD, 2] )
                # print(g2drv)
                # diff = np.matmul(inv(refMat), g2drv)
                diff = ref2g @ g2drv
                # print(idxR, idxD, ref[idxR,:], drv[idxD,:])
                
                dLength = np.hypot(drv[idxD, 0] - ref[idxR, 0], drv[idxD, 1] - ref[idxR, 1])
                # print(diff)
               
                # print(drv.shape[0], " vs. ", ref.shape[0])
                
                if  fabs(dLength) < fabs(cL):
                    # print(g2drv)
                    # print(diff)
                    error_sign = np.sign(diff[1,3])
                    if fabs(error_sign) < 1e-3: error_sign = 1.0
                    cL = dLength * error_sign
                    new_last_idx = copy.deepcopy(idxD)
                    # print("in if:", idxR, new_last_idx, idxD, cL, drv[idxD, 0] - ref[idxR, 0], drv[idxD, 1] - ref[idxR, 1])

            # print(idxR, idxD, ": ", cL)

            lat_error[idxR] = cL

            if fabs(new_last_idx - last_idx) > idx_range / 2: idx_range += min_range
            else : idx_range -= min_range
        
            if idx_range < min_range: idx_range = min_range
            elif idx_range > max_range: idx_range = max_range

            last_idx = copy.deepcopy(new_last_idx)

            # print("new_last_idx = ", new_last_idx)


        return lat_error
                
                



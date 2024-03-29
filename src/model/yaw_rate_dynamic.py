import numpy as np
from scipy import interpolate

class YawRate_Dynamic():
    def __init__(self, model_type="median"):
        # column = velocity
        # row = steer
    
        if model_type == "tractor":
            gain, time_constant, v_unique, steer_angle_unique = self._get_tractor()
        elif model_type == "median":
            gain, time_constant, v_unique, steer_angle_unique = self._get_median()
        elif model_type == "max":
            gain, time_constant, v_unique, steer_angle_unique = self._get_max()
        elif model_type == "min":
            gain, time_constant, v_unique, steer_angle_unique = self._get_min()

        self.gain_interpolate = interpolate.interp2d(v_unique, steer_angle_unique, gain, kind='cubic')
        self.timeConstant_interpolate = interpolate.interp2d(v_unique, steer_angle_unique, time_constant, kind='cubic')

    def get_gain(self, velocity, steer_angle) :
        return self.gain_interpolate(velocity, steer_angle)

    def get_timeConstant(self, velocity, steer_angle) :
        return self.timeConstant_interpolate(velocity, steer_angle)

    def _get_median(self):
        time_constant = np.array([[0.0018732, 0.027239 , 0.053482 , 0.076927 , 0.098071 , 0.11831  ,
                                    0.13866  , 0.15047  , 0.15304  , 0.15757  , 0.16337  ],
                                [0.0026076, 0.026012 , 0.051332 , 0.073391 , 0.093575 , 0.113    ,
                                    0.12121  , 0.13487  , 0.14156  , 0.14836  , 0.15665  ],
                                [0.0026973, 0.026727 , 0.052112 , 0.074661 , 0.095178 , 0.10578  ,
                                    0.11681  , 0.12772  , 0.13919  , 0.14679  , 0.15521  ],
                                [0.0028075, 0.027626 , 0.053176 , 0.075019 , 0.094254 , 0.1062   ,
                                    0.12056  , 0.12963  , 0.13863  , 0.14794  , 0.15337  ],
                                [0.0029647, 0.028966 , 0.054967 , 0.075572 , 0.096232 , 0.11136  ,
                                    0.12145  , 0.13201  , 0.14074  , 0.14729  , 0.15412  ],
                                [0.0032086, 0.030614 , 0.057643 , 0.079499 , 0.099759 , 0.11292  ,
                                    0.12555  , 0.13486  , 0.14205  , 0.14952  , 0.15289  ],
                                [0.0034431, 0.032645 , 0.061406 , 0.083616 , 0.10416  , 0.11906  ,
                                    0.12886  , 0.1384   , 0.14494  , 0.15118  , 0.1527   ],
                                [0.0041115, 0.035234 , 0.066127 , 0.089135 , 0.11079  , 0.12301  ,
                                    0.13268  , 0.14223  , 0.1486   , 0.15257  , 0.15537  ],
                                [0.0048151, 0.040223 , 0.071945 , 0.096305 , 0.11574  , 0.13103  ,
                                    0.13958  , 0.14808  , 0.15116  , 0.15421  , 0.15438  ],
                                [0.0058773, 0.045105 , 0.08     , 0.10549  , 0.1259   , 0.13737  ,
                                    0.14721  , 0.1521   , 0.15576  , 0.15389  , 0.15625  ]])
        gain = np.array([[0.99998, 0.99813, 0.99258, 0.98347, 0.97099, 0.95539, 0.937  ,
                                    0.91616, 0.89323, 0.8686 , 0.84263],
                                [0.9998 , 0.99791, 0.99222, 0.98288, 0.97011, 0.95417, 0.9354 ,
                                    0.91416, 0.89084, 0.86583, 0.8395 ],
                                [0.99922, 0.99719, 0.99108, 0.98106, 0.96739, 0.95041, 0.93049,
                                    0.90807, 0.88358, 0.85744, 0.83007],
                                [0.99811, 0.99582, 0.98897, 0.97778, 0.9626 , 0.94384, 0.92201,
                                    0.89763, 0.8712 , 0.84324, 0.81419],
                                [0.9962 , 0.99354, 0.9856 , 0.9727 , 0.95531, 0.93403, 0.9095 ,
                                    0.88238, 0.85332, 0.82291, 0.79167],
                                [0.99315, 0.98998, 0.98053, 0.96529, 0.94494, 0.92031, 0.89228,
                                    0.86172, 0.82941, 0.79606, 0.76223],
                                [0.98847, 0.9846 , 0.97316, 0.95484, 0.93067, 0.90183, 0.86953,
                                    0.83488, 0.79884, 0.76221, 0.72562],
                                [0.98151, 0.97674, 0.96266, 0.94038, 0.91142, 0.87749, 0.8402 ,
                                    0.80096, 0.76091, 0.72092, 0.68162],
                                [0.97147, 0.96551, 0.94798, 0.92065, 0.88584, 0.84593, 0.80307,
                                    0.75897, 0.71491, 0.67177, 0.63015],
                                [0.95728, 0.94967, 0.9277 , 0.89402, 0.8522 , 0.80555, 0.7568 ,
                                    0.70792, 0.66024, 0.61455, 0.57132]])
        v_unique = np.array([ 0.1,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. ])
        steer_angle_unique = np.array([0.00174533, 0.08726646, 0.17453293, 0.26179939, 0.34906585,
                                          0.43633231, 0.52359878, 0.61086524, 0.6981317 , 0.78539816])
        return gain, time_constant, v_unique, steer_angle_unique

    def _get_min(self):
        time_constant = np.array([[0.00396739, 0.03400923, 0.0622508 , 0.082397  , 0.09369591,
                                    0.09825001, 0.09816913, 0.09510092],
                                [0.00399245, 0.03415721, 0.06241789, 0.08234361, 0.09319234,
                                    0.09747547, 0.09720422, 0.09440065],
                                [0.00407214, 0.03481137, 0.06284327, 0.08187251, 0.09163713,
                                    0.09524981, 0.09460351, 0.09131864],
                                [0.00420674, 0.03589999, 0.06356236, 0.09951827, 0.0891255 ,
                                    0.09147549, 0.09014643, 0.08632992],
                                [0.00445468, 0.03750553, 0.06458068, 0.07966069, 0.08578976,
                                    0.08645632, 0.08388329, 0.07950095],
                                [0.00473047, 0.03969745, 0.06589797, 0.0782749 , 0.08181126,
                                    0.08042578, 0.07647117, 0.07094933],
                                [0.00516457, 0.04260459, 0.06753065, 0.07665291, 0.07728387,
                                    0.07391164, 0.06835821, 0.06195394],
                                [0.00573109, 0.04635454, 0.06948222, 0.07486484, 0.07243101,
                                    0.06690709, 0.06019516, 0.0530951 ],
                                [0.00646356, 0.0511775 , 0.07177266, 0.07289729, 0.06747232,
                                    0.06028537, 0.05270134, 0.04526213],
                                [0.00749469, 0.05724808, 0.07440549, 0.07077611, 0.06254426,
                                    0.0539842 , 0.04597603, 0.03846613]])
        gain = np.array([[0.99985238, 0.98546109, 0.94427382, 0.88345547, 0.80911468,
                            0.73054339, 0.65278365, 0.57923759],
                        [0.99954691, 0.98440912, 0.94141744, 0.87870204, 0.80288436,
                            0.72368866, 0.64579967, 0.57238135],
                        [0.99859088, 0.98117247, 0.93274326, 0.86454449, 0.78454942,
                            0.70337583, 0.62494601, 0.55172965],
                        [0.9968647 , 0.97552854, 0.91804917, 0.84115869, 0.75457051,
                            0.67021037, 0.5906778 , 0.51745236],
                        [0.99416232, 0.96708979, 0.89699186, 0.80698904, 0.71383382,
                            0.62538378, 0.54418533, 0.47062896],
                        [0.99018025, 0.95528557, 0.86911774, 0.7652974 , 0.6635978 ,
                            0.57078089, 0.48781704, 0.41409317],
                        [0.98449494, 0.93933228, 0.83389368, 0.71535352, 0.60547387,
                            0.50897231, 0.42511249, 0.35241283],
                        [0.97654909, 0.91819374, 0.79075221, 0.65786236, 0.54136509,
                            0.4429787 , 0.3601823 , 0.29060203],
                        [0.96559066, 0.89053299, 0.73916526, 0.59373716, 0.47336361,
                            0.3758636 , 0.29677739, 0.23263061],
                        [0.95061257, 0.85466239, 0.67874044, 0.52413094, 0.40362709,
                            0.31034597, 0.23766828, 0.18085065]])
        v_unique = np.array([ 0.1,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7.])
        steer_angle_unique = np.array([0.00174533, 0.08726646, 0.17453293, 0.26179939, 0.34906585,
                                          0.43633231, 0.52359878, 0.61086524, 0.6981317 , 0.78539816])
        return gain, time_constant, v_unique, steer_angle_unique

    def _get_max(self):
        time_constant = np.array([[0.018007, 0.16364 , 0.27423 , 0.36906 , 0.45194 , 0.52041 ,
                                    0.57121 , 0.6084  , 0.62988 , 0.63671 , 0.63863 ],
                                [0.017178, 0.1677  , 0.27586 , 0.37065 , 0.45323 , 0.5211  ,
                                    0.57106 , 0.60718 , 0.62731 , 0.63557 , 0.63403 ],
                                [0.017862, 0.16887 , 0.27733 , 0.37546 , 0.45715 , 0.5232  ,
                                    0.57061 , 0.59911 , 0.62077 , 0.6311  , 0.63094 ],
                                [0.01835 , 0.17096 , 0.28526 , 0.38357 , 0.46371 , 0.52669 ,
                                    0.56986 , 0.59822 , 0.60991 , 0.61528 , 0.61051 ],
                                [0.019191, 0.179   , 0.29661 , 0.3932  , 0.4718  , 0.52862 ,
                                    0.5688  , 0.58554 , 0.60197 , 0.60152 , 0.59349 ],
                                [0.021416, 0.19158 , 0.31027 , 0.40871 , 0.48291 , 0.53478 ,
                                    0.5674  , 0.57462 , 0.58337 , 0.57556 , 0.57132 ],
                                [0.023232, 0.205   , 0.32943 , 0.42687 , 0.49807 , 0.54235 ,
                                    0.56564 , 0.57476 , 0.56984 , 0.5547  , 0.53639 ],
                                [0.02689 , 0.22465 , 0.35464 , 0.44989 , 0.51417 , 0.55137 ,
                                    0.56089 , 0.56124 , 0.54635 , 0.5318  , 0.50641 ],
                                [0.030487, 0.24767 , 0.38582 , 0.47806 , 0.5335  , 0.55819 ,
                                    0.56122 , 0.55031 , 0.52834 , 0.50804 , 0.48144 ],
                                [0.03706 , 0.27833 , 0.42391 , 0.51179 , 0.55774 , 0.56964 ,
                                    0.55834 , 0.53821 , 0.51202 , 0.48363 , 0.44707 ]])
        gain = np.array([[0.99992, 0.99518, 0.98099, 0.95822, 0.92806, 0.89196, 0.8515 ,
                                    0.80822, 0.76353, 0.71838, 0.67374],
                                [0.99988, 0.99496, 0.98038, 0.95703, 0.92618, 0.88936, 0.84822,
                                    0.80431, 0.75901, 0.7135 , 0.66868],
                                [0.99958, 0.99425, 0.97848, 0.95335, 0.92042, 0.88147, 0.83834,
                                    0.79272, 0.74606, 0.69954, 0.65403],
                                [0.99891, 0.99286, 0.97502, 0.9469 , 0.91049, 0.86804, 0.82172,
                                    0.77342, 0.72466, 0.6766 , 0.63008],
                                [0.99758, 0.99047, 0.96963, 0.93719, 0.89591, 0.84871, 0.79817,
                                    0.74643, 0.69506, 0.64518, 0.59754],
                                [0.99523, 0.98663, 0.96169, 0.92353, 0.87603, 0.82296, 0.76746,
                                    0.71184, 0.65768, 0.60599, 0.55737],
                                [0.99131, 0.98071, 0.9504 , 0.90501, 0.84999, 0.79023, 0.72936,
                                    0.66981, 0.61306, 0.55992, 0.51078],
                                [0.98513, 0.97191, 0.93469, 0.88047, 0.81683, 0.74989, 0.68371,
                                    0.62065, 0.56194, 0.50808, 0.4592 ],
                                [0.97582, 0.95913, 0.9132 , 0.84852, 0.77547, 0.70139, 0.63051,
                                    0.56487, 0.50524, 0.45175, 0.40415],
                                [0.96217, 0.94097, 0.88417, 0.80751, 0.72479, 0.6443 , 0.56997,
                                    0.50319, 0.44411, 0.39235, 0.34726]])
        v_unique = np.array([ 0.1,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. ])
        steer_angle_unique = np.array([0.00174533, 0.08726646, 0.17453293, 0.26179939, 0.34906585,
                                          0.43633231, 0.52359878, 0.61086524, 0.6981317 , 0.78539816])
        return gain, time_constant, v_unique, steer_angle_unique

    def _get_tractor(self):
        gain = np.array([[0.9999462 , 0.99474732, 0.97931661, 0.95463529, 0.92210024,
                                0.88339129, 0.84027845, 0.79445644, 0.74742722, 0.7004353 ,
                                0.65444838],
                            [0.99985782, 0.99457047, 0.97892637, 0.95391584, 0.92098783,
                                0.88186857, 0.83836698, 0.7922047 , 0.74489821, 0.69769621,
                                0.65156352],
                            [0.99952337, 0.99400876, 0.97768805, 0.95167696, 0.91756041,
                                0.87720757, 0.83254512, 0.78537438, 0.73725316, 0.68944049,
                                0.6428905 ],
                            [0.9987916 , 0.9928739 , 0.97539781, 0.94767682, 0.91154963,
                                0.86913533, 0.82255968, 0.77375254, 0.72433285, 0.67556888,
                                0.62839053],
                            [0.99739505, 0.99088015, 0.97170625, 0.94150161, 0.9025    ,
                                0.8571945 , 0.80799326, 0.7569948 , 0.7058861 , 0.65593136,
                                0.60801348],
                            [0.99494747, 0.98760446, 0.96610293, 0.93254918, 0.88975822,
                                0.84074317, 0.78827547, 0.73464564, 0.68159534, 0.63035418,
                                0.58172342],
                            [0.99092681, 0.98247621, 0.95789058, 0.92000356, 0.87245829,
                                0.81895678, 0.76270144, 0.70616884, 0.65111287, 0.59867658,
                                0.54953186],
                            [0.98465027, 0.97473462, 0.94614546, 0.90279909, 0.84950484,
                                0.79083715, 0.73046287, 0.67099195, 0.61411015, 0.56079825,
                                0.51153929],
                            [0.97523208, 0.96339531, 0.92965804, 0.87957324, 0.81955976,
                                0.75523617, 0.69069836, 0.62856935, 0.57034227, 0.51673805,
                                0.4679839 ],
                            [0.96154714, 0.94717096, 0.90684671, 0.84861063, 0.78104248,
                                0.71090621, 0.64257213, 0.57846907, 0.51972975, 0.46670384,
                                0.41929666]])
        time_constant = np.array([[0.00922043, 0.1026875 , 0.18293396, 0.25817509, 0.31493038,
                            0.36417902, 0.40355144, 0.43248701, 0.45154765, 0.46181755,
                            0.46836556],
                        [0.0099626 , 0.09740269, 0.18088311, 0.25610559, 0.31836142,
                            0.3643634 , 0.40299135, 0.43105196, 0.4524199 , 0.46241066,
                            0.46836682],
                        [0.0103871 , 0.09917121, 0.18370361, 0.25957708, 0.31933193,
                            0.36805264, 0.40614703, 0.43348747, 0.45084215, 0.463359  ,
                            0.4685699 ],
                        [0.0107446 , 0.10336915, 0.18857901, 0.26260535, 0.32517863,
                            0.37315425, 0.40638992, 0.43620127, 0.45201464, 0.46328758,
                            0.46715232],
                        [0.01135411, 0.10913009, 0.19563391, 0.27054871, 0.33318708,
                            0.37626317, 0.41145681, 0.43529356, 0.45293066, 0.46094771,
                            0.46032402],
                        [0.01195892, 0.11448954, 0.20253284, 0.28111375, 0.33984671,
                            0.38492768, 0.41767509, 0.43877769, 0.45350928, 0.4579633 ,
                            0.45665971],
                        [0.01336102, 0.12404993, 0.21445532, 0.29303306, 0.35298512,
                            0.39572649, 0.42539834, 0.44280505, 0.45382409, 0.45602789,
                            0.45167142],
                        [0.01650981, 0.13286301, 0.23011813, 0.31021427, 0.36686932,
                            0.40524528, 0.43072439, 0.44743061, 0.45395618, 0.45204755,
                            0.44328178],
                        [0.04310396, 0.14966698, 0.25089676, 0.33038597, 0.38576107,
                            0.42088651, 0.44120935, 0.45259721, 0.45381997, 0.44684543,
                            0.43335784],
                        [0.06578827, 0.16711618, 0.27596481, 0.35563624, 0.40630046,
                            0.43553055, 0.45315911, 0.45506763, 0.45287177, 0.44056984,
                            0.42180979]])
        v_unique = np.array([ 0.1,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. ])
        steer_angle_unique = np.array([0.00174533, 0.08726646, 0.17453293, 0.26179939, 0.34906585,
                                          0.43633231, 0.52359878, 0.61086524, 0.6981317 , 0.78539816])
        return gain, time_constant, v_unique, steer_angle_unique






if __name__ == "__main__":

    # yaw_Dynamic = YawRate_Dynamic("tractor")
    # gain, _, v, _ = yaw_Dynamic._get_tractor()
    # yaw_Dynamic = YawRate_Dynamic("median")
    # gain, _, v, _ = yaw_Dynamic._get_median()
    # yaw_Dynamic = YawRate_Dynamic("max")
    # gain, _, v, _ = yaw_Dynamic._get_max()
    yaw_Dynamic = YawRate_Dynamic("min")
    gain, _, v, _ = yaw_Dynamic._get_min()

    import matplotlib.pyplot as plt
    v_inter = np.arange(0.1, 10.0, 0.259)
    steer_inter = np.arange(0.0, 0.6, 0.088)
    # v_inter = 0.9
    # steer_inter = 0.35

    gain_inter = yaw_Dynamic.get_gain(v_inter, steer_inter)
    print(gain_inter)
    print("===> gain_inter = ", gain_inter, "vs.", gain[4, 1])

    print("v_inter.shape: ", v_inter.shape)
    print("gain_inter.shape: ", gain_inter.shape)
    print("v.shape: ", v.shape)
    print("gain.shape: ", gain.shape)

    plt.plot(v, gain[0,:], 'ro-', v_inter, gain_inter[0,:], 'bo-')
    plt.show()

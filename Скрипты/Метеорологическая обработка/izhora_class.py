class izhora_class:

    def __init__(self, prepared_list, station_names, basin_name):
        self.prepared_list = prepared_list
        self.station_names = station_names
        self.basin_name = basin_name

        import subprocess
        import sys
        import os

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

        needed_packages = ['pandas', 'numpy', 'datetime', 'tqdm', 'pymannkendall', 'scipy']

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        for package in needed_packages:
            if package in installed_packages:
                pass
            else:
                install(package)
        
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from tqdm import trange, tqdm
        import pymannkendall as mk
        from scipy import stats

        df_T = self.prepared_list

        def autocorr(x, t=1):
            return np.corrcoef(np.array([x[:-t], x[t:]]))[0][1]

        def running_mean(x, N):
            x = np.array(x)
            cumsum = np.cumsum(np.insert(x, 0, 0))
            t = (cumsum[N:] - cumsum[:-N]) / float(N)
            return t[::N]

        def Hodges(inputdata):
            l_avgs = running_mean(inputdata, 2)
            hl_est = np.nanmedian(l_avgs)
            return hl_est

        def Pettitt2(signal):
            T=len(signal)
            U = [[[np.sign(signal[i] - signal[j]) for j in range(t+1, T)] for i in range(t)] for t in range(T)]
            U_sum = [[sum(U[i][j]) for j in range(len(U[i]))] for i in range(len(U))]
            U_sum = [sum(U_sum[i]) for i in range(len(U_sum))]
            loc = np.argmax(np.abs(U_sum))
            K = max(np.abs(U_sum))
            p = 2 * np.exp(-6*K**2/(T**3+T**2))
            return (loc, p)

        def Buishand(inputdata):
            inputdata_mean = np.mean(inputdata)
            n  = inputdata.shape[0]
            k = range(n)
            Sk = [np.sum(inputdata[0:x+1] - inputdata_mean) for x in k]
            sigma = np.sqrt(np.sum((inputdata-np.mean(inputdata))**2)/(n-1))
            U = np.sum((Sk[0:(n - 2)]/sigma)**2)/(n * (n + 1))
            Ska = np.abs(Sk)
            S = np.max(Ska)
            K = list(Ska).index(S) + 1
            Skk = (Sk/sigma)
            return (K)

        ### Number of observations
        monthes = ['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь', 'Декабрь', 'Среднегодовое']
        time_of_obs = [pd.Series([len(df_T[i][j]) for j in monthes], index = monthes) for i in range(len(df_T))]
        missings = [pd.Series([sum(df_T[i][j].isna()) for j in monthes], index = monthes) for i in range(len(df_T))]
        actual_obs = [pd.Series([time_of_obs[i][j] - missings[i][j] for j in range(len(time_of_obs[i]))], index = monthes) for i in range(len(time_of_obs))]

        #mean
        MEAN_T = [df_T[i][monthes].mean(axis = 0) 
                for i in trange(len(df_T), desc='AVG calc')]
        #cv
        CV_T = [df_T[i][monthes].std(axis = 0) / df_T[i][monthes].mean(axis = 0)
            for i in trange(len(df_T), desc='CV calc')]

        import pymannkendall as mk # https://github.com/mmhs013/pyMannKendall

        #mk
        MK_T_005 = [[np.array(mk.original_test(df_T[i][j], alpha=0.05))[[0]] if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN] for j in monthes] for i in trange(len(df_T), desc='MK calc 0.05')]

        MK_T_01 = [[np.array(mk.original_test(df_T[i][j], alpha=0.1))[[0, 2, 4]] if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN, np.NaN, np.NaN] for j in monthes] for i in trange(len(df_T), desc='MK calc 0.1')]

        MK_T_trend_005 = [pd.Series([MK_T_005[i][j][0] for j in range(len(MK_T_005[i]))], index = monthes) for i in trange(len(MK_T_005), desc='MK trend 0.05')]

        MK_T_trend_005 =[pd.Series(['0' if MK_T_trend_005[i][j] == 'no trend' else '1' if MK_T_trend_005[i][j] == 'increasing' 
                                    else '-1' for j in range(len(MK_T_trend_005[i]))], index = monthes) for i in range(len(MK_T_trend_005))]


        MK_T_trend_01 = [pd.Series([MK_T_01[i][j][0] for j in range(len(MK_T_01[i]))], index = monthes) for i in trange(len(MK_T_01), desc='MK trend 0.1')]

        MK_T_trend_01 =[pd.Series(['0' if MK_T_trend_01[i][j] == 'no trend' else '1' if MK_T_trend_01[i][j] == 'increasing' 
                                    else '-1' for j in range(len(MK_T_trend_01[i]))], index = monthes) for i in range(len(MK_T_trend_01))]

        MK_T_p = [pd.Series([MK_T_01[i][j][1] for j in range(len(MK_T_01[i]))], index = monthes) for i in trange(len(MK_T_01), desc='MK p')]

        MK_T_tau = [pd.Series([MK_T_01[i][j][2] for j in range(len(MK_T_01[i]))], index = monthes) for i in trange(len(MK_T_01), desc='MK tau')]

        #spearman https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html


        SRC_T = [[stats.spearmanr(df_T[i][j].dropna(), range(len(df_T[i][j].dropna()))) if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN, np.NaN] for j in monthes] for i in trange(len(df_T), desc='SRC calc')]

        SRC_T_rho = [pd.Series([SRC_T[i][j][0] for j in range(len(SRC_T[i]))], index = monthes) for i in trange(len(SRC_T), desc='SRC rho')]

        SRC_T_rho_p = [pd.Series([SRC_T[i][j][1] for j in range(len(SRC_T[i]))], index = monthes) for i in trange(len(SRC_T), desc='SRC p')]

        #THEIL-Sen https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.mstats.theilslopes.html

        THEIL_T = [[stats.theilslopes(df_T[i][j].dropna(), range(len(df_T[i][j].dropna())), 0.90) if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN] for j in monthes] for i in trange(len(df_T), desc='THEIL calc')]

        THEIL_T_S = [pd.Series([THEIL_T[i][j][0] for j in range(len(THEIL_T[i]))], index = monthes) for i in trange(len(THEIL_T), desc='THEIL SLOPE')]

        #AR1

        AR1_T = [pd.Series([autocorr(df_T[i][j].dropna()) if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else np.NaN for j in monthes], index = monthes) for i in trange(len(df_T), desc='AR calc')]

        #TFPW

        TFPW_T_005 = [[np.array(mk.trend_free_pre_whitening_modification_test(df_T[i][j], alpha=0.05))[[0]] if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN] 
                    for j in monthes] for i in trange(len(df_T), desc='TFPW calc 0.05')]

        TFPW_T_01 = [[np.array(mk.trend_free_pre_whitening_modification_test(df_T[i][j], alpha=0.1))[[0, 2, 4, 7]] if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN]*4 
                    for j in monthes] for i in trange(len(df_T), desc='TFPW calc 0.1')]

        TFPW_T_trend_01 = [pd.Series([TFPW_T_01[i][j][0] for j in range(len(TFPW_T_01[i]))], index = monthes) for i in trange(len(TFPW_T_01), desc='TFPW_T trend')]

        TFPW_T_trend_01 =[pd.Series(['0' if TFPW_T_trend_01[i][j] == 'no trend' else '1' if TFPW_T_trend_01[i][j] == 'increasing' 
                                    else '-1' for j in range(len(TFPW_T_trend_01[i]))], index = monthes) for i in range(len(TFPW_T_trend_01))]

        TFPW_T_trend_005 = [pd.Series([TFPW_T_005[i][j][0] for j in range(len(TFPW_T_005[i]))], index = monthes) for i in trange(len(TFPW_T_005), desc='TFPW_T trend')]

        TFPW_T_trend_005 =[pd.Series(['0' if TFPW_T_trend_005[i][j] == 'no trend' else '1' if TFPW_T_trend_005[i][j] == 'increasing' 
                                    else '-1' for j in range(len(TFPW_T_trend_005[i]))], index = monthes) for i in range(len(TFPW_T_trend_005))]

        TFPW_T_p = [pd.Series([TFPW_T_01[i][j][1] for j in range(len(TFPW_T_01[i]))], index = monthes) for i in trange(len(TFPW_T_01), desc='TFPW_T p')]

        TFPW_T_tau = [pd.Series([TFPW_T_01[i][j][2] for j in range(len(TFPW_T_01[i]))], index = monthes) for i in trange(len(TFPW_T_01), desc='TFPW_T tau')]

        TFPW_T_S = [pd.Series([TFPW_T_01[i][j][3] for j in range(len(TFPW_T_01[i]))], index = monthes, dtype = 'float64') for i in trange(len(TFPW_T_01), desc='TFPW_T SLOPE')]

        # Pettitt

        Pettitt_T = [[np.array(Pettitt2(df_T[i][j].dropna().to_numpy())) if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else [np.NaN, np.NaN] for j in monthes] 
                for i in trange(len(df_T), desc='Pettitt calc') if len(df_T[i]) > 2]

        Pettitt_T_pos = [pd.Series([Pettitt_T[i][j][0] for j in range(len(Pettitt_T[i]))], index = monthes) for i in trange(len(Pettitt_T), desc='Pettitt_T pos')]
        Pettitt_T_posY = [pd.Series([df_T[j].date[i] for i in Pettitt_T_pos[j]], index = monthes) 
                            for j in trange(len(Pettitt_T_pos), desc = 'YEAR OF PETTITT')]

        Pettitt_T_p = [pd.Series([Pettitt_T[i][j][1] for j in range(len(Pettitt_T[i]))], index = monthes) for i in trange(len(Pettitt_T), desc='Pettitt_T p')]

        Pettitt_T_005 = [pd.Series(['1' if Pettitt_T_p[j][i] > 0.05 else '0' for i in range(len(Pettitt_T_p[j]))], index = monthes) for j in range(len(Pettitt_T_p))]

        Pettitt_T_01 = [pd.Series(['1' if Pettitt_T_p[j][i] > 0.1 else '0' for i in range(len(Pettitt_T_p[j]))], index = monthes) for j in range(len(Pettitt_T_p))]

        # Buishand

        Buishand_T = [pd.Series([Buishand(df_T[i][j].dropna().to_numpy()) if sum(df_T[i][j].isna()) < len(df_T[i][j]) - 2 else np.NaN for j in monthes], index = monthes) for i in trange(len(df_T), desc='Buishand calc')]
        Buishand_TY = [pd.Series([df_T[j].date[i] for i in Buishand_T[j]], index = monthes)  for j in trange(len(Buishand_T), desc = 'YEAR OF BUISHAND')]

        
        # Hodges-Lehmann Estimator

        Hodges_T = [pd.Series([Hodges(df_T[i][j].dropna().to_numpy()) for j in monthes], index = monthes) for i in trange(len(df_T), desc='Hodges calc')]

        # Change in data

        CHANGE = [pd.Series([THEIL_T_S[i][j] * actual_obs[i][j] if np.abs(AR1_T[i][j]) < 0.20 else TFPW_T_S[i][j] * actual_obs[i][j] for j in range(len(actual_obs[i]))], index = monthes) for i in trange(len(actual_obs), desc = 'CHANGE CALC')]

        CHANGE_percent = [pd.Series([THEIL_T_S[i][j] * 100 * actual_obs[i][j]/abs(MEAN_T[i][j]) if np.abs(AR1_T[i][j]) < 0.20
                            else TFPW_T_S[i][j] * 100 * actual_obs[i][j]/abs(MEAN_T[i][j]) for j in range(len(actual_obs[i]))], index = monthes) for i in trange(len(actual_obs), desc = 'CHANGE PERCENT CALC')]

        index_names = ['Start', 'End',
                'N_obs', 'N_miss', 'Obs',
                'Mean', #'CV',
                'MK_trend_005', 'MK_trend_01', #'MK_p', 'MK_tau',
                #'SRC_rho', 'SRC_rho_p',
                'THEIL_S',
                #'AR1', 
                #'TFPW_trend_005', 'TFPW_trend_01', 'TFPW_p', 'TFPW_tau','TFPW_S',
                #'Pettitt_Y', 'Pettitt_p', 'Pettitt_005', 'Pettitt_01',
                #'Buishand_Y',
                #'Hodges', 
                'Change', 'Change_proc']

        start_year = [pd.Series(df_T[i].date.min(),index = monthes) for i in range(len(df_T))]
        last_year = [pd.Series(df_T[i].date.max(),index = monthes) for i in range(len(df_T))]
        time_of_obs = [pd.Series([len(df_T[i][j]) for j in monthes], index = monthes) for i in range(len(df_T))]
        missings = [pd.Series([sum(df_T[i][j].isna()) for j in monthes], index = monthes) for i in range(len(df_T))]
        observed_gauges_u = self.station_names

        test = [pd.concat([start_year[i], last_year[i],
                            time_of_obs[i], missings[i], actual_obs[i], 
                            MEAN_T[i].astype('float64').apply('{:.3f}'.format), #CV_T[i].astype('float64').apply('{:.2f}'.format),
                            MK_T_trend_005[i], MK_T_trend_01[i], #MK_T_p[i].astype('float64').apply('{:.3f}'.format), MK_T_tau[i].astype('float64').apply('{:.2f}'.format), 
                            #SRC_T_rho[i].astype('float64').apply('{:.2f}'.format), SRC_T_rho_p[i].astype('float64').apply('{:.3f}'.format), 
                            THEIL_T_S[i].astype('float64').apply('{:.3f}'.format), 
                            #AR1_T[i].astype('float64').apply('{:.2f}'.format), 
                            #TFPW_T_trend_005[i], TFPW_T_trend_01[i], TFPW_T_p[i].astype('float64').apply('{:.3f}'.format), TFPW_T_tau[i].astype('float64').apply('{:.2f}'.format), TFPW_T_S[i].astype('float64').apply('{:.3f}'.format),
                            #Pettitt_T_posY[i], Pettitt_T_p[i].astype('float64').apply('{:.3f}'.format), Pettitt_T_005[i], Pettitt_T_01[i],
                            #Buishand_TY[i],
                            #Hodges_T[i].astype('float64').apply('{:.2f}'.format), 
                            CHANGE[i].astype('float64').apply('{:.2f}'.format), CHANGE_percent[i].astype('float64').apply('{:.3}'.format)], axis = 1).T for i in trange(len(df_T))]

        test = [test[i].set_index([index_names]) for i in range(len(test))]

        FOR_WHAT = self.basin_name

        path_for_data = os.getcwd() + r'\STAT\{}'.format(FOR_WHAT) + '_stat'

        if not os.path.exists(path_for_data):
            os.makedirs(path_for_data)

        for i in trange(len(test)):
            test[i].to_csv(path_for_data + '\{}.txt'.format(observed_gauges_u[i]), sep="\t", index = True)
            
        tryout = pd.concat(test)

        tryout.to_csv(path_for_data + '\MEGA_FILE.txt', sep = '\t', index = True)

        test_trend = [test[k].iloc[np.r_[7:9, 15:17]] for k in range(len(test))]

        MEGA_trend = pd.concat(test_trend)

        MEGA_trend.to_csv(path_for_data + '\MEGA_TREND.txt', sep = '\t', index = True)

        test_change = [test[k].iloc[np.r_[-2]] for k in range(len(test))]

        MEGA_change = pd.concat(test_change)

        MEGA_change.to_csv(path_for_data + '\MEGA_change.txt', sep = '\t', index = True)

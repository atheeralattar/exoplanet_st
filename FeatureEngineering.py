import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import GatherData as gd
import sys
import astropy as a
import json

sys.path.insert(0, './ExtraPackages')
#import raDec2Pix
import tess_stars2px as tess
import time as t
from scipy import interpolate
from astropy import constants as const
import requests
import csv
from astroquery.mast import Observations
import xmltodict
import os


def FoldLightKurveData(LightKurveObj, Period, EpochStartTime):
    """
    HelperFunction to fold lightkurve data
    """
    return LightKurveObj.fold(period=Period, epoch_time=EpochStartTime)


def StichObservations(LightKurveObj, mission):
    """
    Helper Function to stich lightkurve obs together
    """
    if (mission == 'Kepler'):
        return lk.LightCurveCollection.stitch(LightKurveObj)
    elif (mission == 'TESS'):
        return StichTESS(LightKurveObj)
    else:
        print('Please Choose valid Mission. Recall K2 is not valid.')
        return None


def StichTESS(obs):
    """
    Helper function to stich TESS LC data togehter
    """
    for i in range(0, len(obs)):
        if (i == 0):
            temp = lk.LightCurve(obs[0])
        elif (i == 1):
            temp = lk.LightCurveCollection.stitch([temp, obs[1]])
        else:
            try:
                temp = lk.LightCurveCollection.stitch([temp, obs[i]])
            except:
                pass

    return temp


def FullOrbitView(FoldedLightKurveObj, period, Duration, Centroid=False, featurename='FullOrbitView',
                  flux_str='WhitenedFlux', ID=None, mission=None, n_bins=301, Plot=False):
    """
    Creates Full Orbit view
    """
    bin_width = period / n_bins  # period is in days ,duration is in hours
    data = FoldedLightKurveObj.bin(time_bin_size=bin_width, n_bins=n_bins)
    if (Plot):
        data.scatter()
        plt.savefig(mission + '_' + str(ID) + '_' + featurename + '.png')
        plt.close()
    if (Centroid == False):
        flux = data[flux_str].value.data
        phase = data['time'].value.data
    else:
        flux = data[flux_str].value
        phase = data['time'].value
    return {'phase': phase, 'flux': flux}


def OddEvenFluxFeatures(FoldedLightKurveObj, Duration, ID, mission, n_bins=31, Plot=False):
    """
    Creates Odd and Even Flux features
    """
    tv_durr = 5 * Duration / 24  # Duration is in hours #see pg 14 ExoMiner Algo2
    T_window = np.abs(tv_durr / 2)
    bin_width = Duration / 24 * .16
    odd_data = FoldedLightKurveObj[FoldedLightKurveObj.odd_mask].bin(time_bin_size=bin_width, time_bin_start=-T_window,
                                                                     n_bins=n_bins)
    even_data = FoldedLightKurveObj[FoldedLightKurveObj.even_mask].bin(time_bin_size=bin_width,
                                                                       time_bin_start=-T_window,
                                                                       n_bins=n_bins)
    if (Plot):
        odd_data.scatter()
        plt.savefig(mission + '_' + str(ID) + '_OddTransitView.png')
        plt.close()
        even_data.scatter()
        plt.savefig(mission + '_' + str(ID) + '_EvenTransitView.png')
        plt.close()
    oddflux = odd_data['flux'].value
    oddphase = odd_data['time'].value
    evenflux = odd_data['flux'].value
    evenphase = odd_data['time'].value
    return ({'phase': oddphase, 'flux': oddflux}, {'phase': evenphase, 'flux': evenflux})


def PadRanges(padding, transit_vals):
    """
    Generates ranges for padding for Algo 1
    """
    r = np.zeros((len(transit_vals), 2))
    for i in range(0, len(transit_vals)):
        r[i, :] = [transit_vals[i] - padding / 2, transit_vals[i] + padding / 2]
    return r


def Algo1Flux(LightKurveObj, Paramters):
    """
    Algo 1 Computation for Flux
    """
    data = LightKurveObj.remove_nans()
    data = data.normalize(unit='unscaled')
    time_diff = np.append(0, np.diff(data.time.value))
    number_of_occurences = int(len(time_diff[time_diff > .75]))
    num_arrays = number_of_occurences + 1
    arr_idxs = np.append(np.append(0, np.where(time_diff > .75)[0]), len(data.time.value))
    adjusted_padding = min(3 * Paramters['Duration[hrs]'] / 24, Paramters['Period[Days]'])
    transit_vals = np.append(Paramters['FirstEpoch'] + Paramters['FirstEpoch'] * np.linspace(0, 50, 50)
                             , Paramters['FirstEpoch'] - Paramters['FirstEpoch'] * np.linspace(1, 50, 49))
    transit_vals = transit_vals[
        np.logical_and(transit_vals >= data.time.value.min(), transit_vals <= data.time.value.max())]
    F = np.zeros(len(data.flux.value))
    for j in range(0, num_arrays):
        time_v = data.time.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        flux_v = data.flux.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        transit_vals_v = transit_vals[np.logical_and(transit_vals >= time_v.min(), transit_vals < time_v.max())]
        Ranges = PadRanges(adjusted_padding, transit_vals_v)
        for i in range(0, len(Ranges)):
            low = (time_v > Ranges[i, 0])
            high = (time_v < Ranges[i, 1])
            cond = np.logical_and(low, high)
            if (np.any(cond) == True):
                x = time_v[~cond]
                y = flux_v[~cond]
                f = interpolate.interp1d(x, y)
                time_vals_removed = time_v[cond]
                computed_values = f(time_vals_removed)
                flux_v[cond] = computed_values
            else:
                pass

        try:
            spline = interpolate.UnivariateSpline(time_v, flux_v)
            f_add = flux_v / spline(time_v)
        except:
            tck, u = interpolate.splprep([time_v, flux_v])
            spline = interpolate.splev(u, tck)
            f_add = flux_v / spline[0]

        F[int(arr_idxs[j]):int(arr_idxs[j + 1])] = f_add
    data['WhitenedFlux'] = F
    return data


def Algo1FluxSecondary(LightKurveObj, Paramters):
    """
    Algo 1 computation for secondary obj
    """
    data = LightKurveObj.remove_nans()
    data = data.normalize(unit='unscaled')
    time_diff = np.append(0, np.diff(data.time.value))
    number_of_occurences = int(len(time_diff[time_diff > .75]))
    num_arrays = number_of_occurences + 1
    arr_idxs = np.append(np.append(0, np.where(time_diff > .75)[0]), len(data.time.value))
    adjusted_padding = 3 * Paramters['Duration[hrs]'] / 24
    transit_vals = np.append(Paramters['FirstEpoch'] + Paramters['FirstEpoch'] * np.linspace(0, 50, 50)
                             , Paramters['FirstEpoch'] - Paramters['FirstEpoch'] * np.linspace(1, 50, 49))
    transit_vals = transit_vals[
        np.logical_and(transit_vals >= data.time.value.min(), transit_vals <= data.time.value.max())]
    F = np.zeros(len(data.flux.value))
    for j in range(0, num_arrays):
        time_v = data.time.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        flux_v = data.flux.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        transit_vals_v = transit_vals[np.logical_and(transit_vals >= time_v.min(), transit_vals < time_v.max())]
        Ranges = PadRanges(adjusted_padding, transit_vals_v)
        for i in range(0, len(Ranges)):
            low = (time_v > Ranges[i, 0])
            high = (time_v < Ranges[i, 1])
            cond = np.logical_and(low, high)
            if (np.any(cond) == True):
                x = time_v[~cond]
                y = flux_v[~cond]
                f = interpolate.interp1d(x, y)
                time_vals_removed = time_v[cond]
                computed_values = f(time_vals_removed)
                flux_v[cond] = computed_values
            else:
                pass
        try:
            spline = interpolate.UnivariateSpline(time_v, flux_v)
            f_add = flux_v / spline(time_v)
        except:
            tck, u = interpolate.splprep([time_v, flux_v])
            spline = interpolate.splev(u, tck)
            f_add = flux_v / spline[0]
        F[int(arr_idxs[j]):int(arr_idxs[j + 1])] = f_add
    data['WhitenedFlux'] = F
    return data


def Algo1Centroid(LightKurveObj, Paramters):
    """
    Algo q compuatation for Centroid data
    """
    data = LightKurveObj.remove_nans()
    time_diff = np.append(0, np.diff(data.time.value))
    number_of_occurences = int(len(time_diff[time_diff > .75]))
    num_arrays = number_of_occurences + 1
    arr_idxs = np.append(np.append(0, np.where(time_diff > .75)[0]), len(data.time.value))
    adjusted_padding = min(3 * Paramters['Duration[hrs]'] / 24, Paramters['Period[Days]'])
    transit_vals = np.append(Paramters['FirstEpoch'] + Paramters['FirstEpoch'] * np.linspace(0, 50, 50)
                             , Paramters['FirstEpoch'] - Paramters['FirstEpoch'] * np.linspace(1, 50, 49))
    transit_vals = transit_vals[
        np.logical_and(transit_vals >= data.time.value.min(), transit_vals <= data.time.value.max())]
    F_ra = np.zeros(len(data.RA.value))
    F_dec = np.zeros(len(data.DEC.value))
    for j in range(0, num_arrays):
        time_v = data.time.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        RA_v = data.RA.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        DEC_v = data.DEC.value[int(arr_idxs[j]):int(arr_idxs[j + 1])]
        transit_vals_v = transit_vals[np.logical_and(transit_vals >= time_v.min(), transit_vals < time_v.max())]
        Ranges = PadRanges(adjusted_padding, transit_vals_v)
        for i in range(0, len(Ranges)):
            low = (time_v > Ranges[i, 0])
            high = (time_v < Ranges[i, 1])
            cond = np.logical_and(low, high)
            if (np.any(cond) == True):
                x = time_v[~cond]
                y_ra = RA_v[~cond]
                y_dec = DEC_v[~cond]
                f_ra = interpolate.interp1d(x, y_ra)
                f_dec = interpolate.interp1d(x, y_dec)
                time_vals_removed = time_v[cond]
                computed_values_ra = f_ra(time_vals_removed)
                computed_values_dec = f_dec(time_vals_removed)
                RA_v[cond] = computed_values_ra
                DEC_v[cond] = computed_values_dec
            else:
                pass

        try:
            spline_ra = interpolate.UnivariateSpline(time_v, RA_v)
            spline_dec = interpolate.UnivariateSpline(time_v, DEC_v)
            f_add_ra = RA_v / spline_ra(time_v)
            f_add_dec = RA_v / spline_dec(time_v)
        except:
            tck_ra, u_ra = interpolate.splprep([time_v, RA_v])
            tck_dec, u_dec = interpolate.splprep([time_v, DEC_v])
            spline_ra = interpolate.splev(u_ra, tck_ra)
            spline_dec = interpolate.splev(u_dec, tck_dec)
            f_add_ra = spline_ra[0]
            f_add_dec = spline_dec[0]
        F_ra[int(arr_idxs[j]):int(arr_idxs[j + 1])] = f_add_ra
        F_dec[int(arr_idxs[j]):int(arr_idxs[j + 1])] = f_add_dec
    data['WhitenedRA'] = F_ra
    data['WhitenedDEC'] = F_dec
    return data


def CentroidFeatures(FoldedLightKurveObj, TransitDepth, Period, Duration, ID=None, mission=None, Plot=False):
    """
    Generates Centroid Features
    """
    C_RA_out, C_DEC_out = AverageOutOftransitCentroidPosition(FoldedLightKurveObj, Duration)
    C_RA_n = FoldedLightKurveObj['WhitenedRA'] * C_RA_out
    C_DEC_n = FoldedLightKurveObj['WhitenedDEC'] * C_DEC_out
    dec_target = FoldedLightKurveObj['DEC']
    ra_target = FoldedLightKurveObj['RA']
    RA_transit = C_RA_out - (1 / TransitDepth - 1) * (C_RA_n - C_RA_out) / np.cos(dec_target * np.pi / 180)
    DEC_transit = C_DEC_out - (1 / TransitDepth - 1) * (C_DEC_n - C_DEC_out)
    delta_ra = (RA_transit - ra_target) * np.cos(dec_target * np.pi / 180)
    delta_dec = (DEC_transit - dec_target)
    D = np.sqrt(delta_ra ** 2 + delta_dec ** 2)
    D_arcsec = D / 3600
    FoldedLightKurveObj['CentroidMotion[arcsec]'] = D_arcsec
    FoldedLightKurveObj = FoldedLightKurveObj.select_flux('CentroidMotion[arcsec]')
    CentroidFullOrbitData = FullOrbitView(FoldedLightKurveObj, Centroid=True, featurename='CentroidFullOrbitView',
                                          flux_str='CentroidMotion[arcsec]', period=Period,
                                          Duration=Duration, ID=ID, mission=mission,
                                          Plot=Plot)
    CentroidTransitOrbitData = TransitView(FoldedLightKurveObj, Centroid=True, featurename='CentroidTransitOrbitView',
                                           flux_str='CentroidMotion[arcsec]', Duration=Duration, ID=ID,
                                           mission=mission,
                                           Plot=Plot)
    return CentroidFullOrbitData, CentroidTransitOrbitData


def GenFeatures(FoldedLightKurveObj, SecondaryLightKurveObj, Parameters, Plot=False):
    """
    Generates all features for the model.
    """
    Period = Parameters['Period[Days]']
    Duration = Parameters['Duration[hrs]']
    ID = Parameters['ID']
    mish = Parameters['Mission']

    TransitDepth = Parameters['TransitDepth[ppm]']
    #################### Feature Branches 1 and 2 complete
    FullOrbitData = FullOrbitView(FoldedLightKurveObj, period=Period, Duration=Duration, ID=ID, mission=mish,
                                  Plot=Plot)
    Scalar_for_branch2 = TransitDepth
    TransitViewData = TransitView(FoldedLightKurveObj, Duration=Duration, ID=ID, mission=mish,
                                  Plot=Plot)
    #####################################################
    #################### Feature Branch 5 complete
    OddData, EvenData = OddEvenFluxFeatures(FoldedLightKurveObj, Duration=Duration, ID=ID, mission=mish,
                                            Plot=Plot)
    #####################################################
    ####### Feature Branches 3 and 4 Complete
    CentroidFullOrbitData, CentroidTransitOrbitData = CentroidFeatures(FoldedLightKurveObj, TransitDepth=TransitDepth,
                                                                       Period=Period, Duration=Duration, ID=ID,
                                                                       mission=mish, Plot=Plot)
    #####################################################
    ##### Feature Branch 6 Complete
    SecondaryTransitViewData = TransitView(SecondaryLightKurveObj, featurename='SecondaryTransitView',
                                           Duration=Duration, ID=ID, mission=mish,
                                           Plot=Plot)

    StellarParamsScalars = np.array([Parameters['SV1_TEFF'],
                                     Parameters['SV2_StellarRadius'],
                                     Parameters['SV3_StellarMass'],
                                     Parameters['SV4_StellarDensity'],
                                     Parameters['SV5_SurfaceGravity'],
                                     Parameters['SV6_Metalicity']])
    StellarParamsScalars = np.nan_to_num(StellarParamsScalars, nan=-1)

    CentroidScalars = np.array([Parameters['Centroid1_RA_OOT'],
                                Parameters['Centroid2_deltaRA_OOT'],
                                Parameters['Centroid3_DEC_OOT'],
                                Parameters['Centroid4_deltaDEC_OOT'],
                                Parameters['Centroid5_CentroidOffset']])
    CentroidScalars = np.nan_to_num(CentroidScalars, nan=-1)
    SecondaryScalars = np.array([Parameters['Secondary1_GeoAlbedoComp'],
                                 Parameters['Secondary2_PlanetTemp'],
                                 Parameters['Secondary3_MaxMES'],
                                 Parameters['Secondary4_WeakSecondaryDepth']])

    SecondaryScalars = np.nan_to_num(SecondaryScalars, nan=-1)
    DVDiagnosticScalars = np.array([Parameters['DV1_OpticalGhostCore'],
                                    Parameters['DV2_OpticalGhostHalo'],
                                    Parameters['DV3_BootstrapFalseAlarmProb'],
                                    Parameters['DV4_RollingBandHist'],
                                    Parameters['DV5_OrbitalPeriod'],
                                    Parameters['DV6_PlanetRadius']])
    DVDiagnosticScalars = np.nan_to_num(DVDiagnosticScalars, nan=-1)

    Data = {'FullOrbitData': FullOrbitData['flux'],
            'TransitViewData': TransitViewData['flux'],
            'TransitDepth': TransitDepth,
            'CentroidFullOrbitData': CentroidFullOrbitData['flux'],
            'CentroidTransitOrbitData': CentroidTransitOrbitData['flux'],
            'CentroidScalars': CentroidScalars,
            'OddData': OddData['flux'],
            'EvenData': EvenData['flux'],
            'SecondaryTransitViewData': SecondaryTransitViewData['flux'],
            'SecondaryScalars': SecondaryScalars,
            'StellarParamsScalars': StellarParamsScalars,
            'DVDiagnosticScalars': DVDiagnosticScalars
            }
    return Data


def TESSAstroQuerry(Params):
    """
    Submits quereys to MAST using astroquerey to find TESS Scalar featuers......... very slow :(
    """
    ID = Params['ID']
    obs = Observations.query_criteria(obs_collection=Params['Mission'], target_name=ID[4:])
    temp = Observations.get_product_list(obs)
    temp = Observations.filter_products(temp, extension=['.xml'])
    chosen = temp[0]
    Observations.download_products(chosen)
    with open('./mastDownload/' + Params['Mission'] + '/' + chosen['obs_id'] + '/' + chosen['dataURI'][18:], 'r',
              encoding='utf-8') as file:
        dv = file.read()
    dv = xmltodict.parse(dv)
    try:
        Params['DV1_OpticalGhostCore'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:ghostDiagnosticResults'][
                'dv:coreApertureCorrelationStatistic']['@value'])
        Params['DV2_OpticalGhostHalo'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:ghostDiagnosticResults'][
                'dv:haloApertureCorrelationStatistic']['@value'])
        Params['DV3_BootstrapFalseAlarmProb'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:bootstrapResults']['@bootstrapThresholdForDesiredPfa'])
        Params['DV4_RollingBandHist'] = float(-1)
        Params['DV5_OrbitalPeriod'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:planetCandidate']['@orbitalPeriodInDays'])
        Params['DV6_PlanetRadius'] = float(dv['dv:dvTargetResults']['dv:radius']['@value'])
        Params['Secondary1_GeoAlbedoComp'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:secondaryEventResults']['dv:comparisonTests'][
                'dv:albedoComparisonStatistic']['@value'])
        Params['Secondary2_PlanetTemp'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:secondaryEventResults']['dv:planetParameters'][
                'dv:planetEffectiveTemp']['@value'])
        Params['Secondary3_MaxMES'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:planetCandidate']['dv:weakSecondary']['@maxMes'])
        Params['Secondary4_WeakSecondaryDepth'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:planetCandidate']['dv:weakSecondary']['dv:depthPpm'][
                '@value'])
        Params['Centroid1_RA_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanRaOffset']['@value'])
        Params['Centroid2_deltaRA_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanRaOffset']['@uncertainty'])
        Params['Centroid3_DEC_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanDecOffset']['@value'])
        Params['Centroid4_deltaDEC_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanDecOffset']['@uncertainty'])
        Params['Centroid5_CentroidOffset'] = float(
            dv['dv:dvTargetResults']['dv:planetResults'][0]['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msControlCentroidOffsets']['dv:meanSkyOffset']['@value'])
    except:
        Params['DV1_OpticalGhostCore'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:ghostDiagnosticResults'][
                'dv:coreApertureCorrelationStatistic']['@value'])
        Params['DV2_OpticalGhostHalo'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:ghostDiagnosticResults'][
                'dv:haloApertureCorrelationStatistic']['@value'])
        Params['DV3_BootstrapFalseAlarmProb'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:bootstrapResults']['@bootstrapThresholdForDesiredPfa'])
        Params['DV4_RollingBandHist'] = float(-1)
        Params['DV5_OrbitalPeriod'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:planetCandidate']['@orbitalPeriodInDays'])
        Params['DV6_PlanetRadius'] = float(dv['dv:dvTargetResults']['dv:radius']['@value'])
        Params['Secondary1_GeoAlbedoComp'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:secondaryEventResults']['dv:comparisonTests'][
                'dv:albedoComparisonStatistic']['@value'])
        Params['Secondary2_PlanetTemp'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:secondaryEventResults']['dv:planetParameters'][
                'dv:planetEffectiveTemp']['@value'])
        Params['Secondary3_MaxMES'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:planetCandidate']['dv:weakSecondary']['@maxMes'])
        Params['Secondary4_WeakSecondaryDepth'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:planetCandidate']['dv:weakSecondary']['dv:depthPpm'][
                '@value'])
        Params['Centroid1_RA_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanRaOffset']['@value'])
        Params['Centroid2_deltaRA_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanRaOffset']['@uncertainty'])
        Params['Centroid3_DEC_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanDecOffset']['@value'])
        Params['Centroid4_deltaDEC_OOT'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msTicCentroidOffsets']['dv:meanDecOffset']['@uncertainty'])
        Params['Centroid5_CentroidOffset'] = float(
            dv['dv:dvTargetResults']['dv:planetResults']['dv:centroidResults']['dv:differenceImageMotionResults'][
                'dv:msControlCentroidOffsets']['dv:meanSkyOffset']['@value'])
    os.remove('./mastDownload/' + Params['Mission'] + '/' + chosen['obs_id'] + '/' + chosen['dataURI'][18:])
    os.rmdir('./mastDownload/' + Params['Mission'] + '/' + chosen['obs_id'])
    os.rmdir('./mastDownload/' + Params['Mission'])
    return Params


def OpenSuplamental(Params):
    """
    Helper Function to find scalar params for TESS and Kepler.
    """
    if (Params['Mission'] == 'TESS'):
        data = pd.read_csv('./data/exofop_tess_tois.csv', header=1)
        ID_striped = Params['ID'][4:]
        ID_Data = data[data['TIC ID'] == int(ID_striped)]
        Params['SV3_StellarMass'] = ID_Data['Stellar Mass (M_Sun)'].values[0]
        Params['SV6_Metalicity'] = ID_Data['Stellar Metallicity'].values[0]
        # g/cm3 = ([M_sun] * [g/M_sun]) /(4 pi ([R_sun]*[cm/R_sun])**3)
        Mass = ID_Data['Stellar Mass (M_Sun)'].values[0] * const.M_sun.to('g').value
        Volume = 4 * np.pi * (ID_Data['Stellar Radius (R_Sun)'].values[0] * const.R_sun.to('cm').value) ** 3
        Params['SV4_StellarDensity'] = Mass / Volume
        try:
            Params = TESSAstroQuerry(Params)
        except:
            Params['DV1_OpticalGhostCore'] = -1
            Params['DV2_OpticalGhostHalo'] = -1
            Params['DV3_BootstrapFalseAlarmProb'] = -1
            Params['DV4_RollingBandHist'] = -1
            Params['DV5_OrbitalPeriod'] = -1
            Params['DV6_PlanetRadius'] = -1
            Params['Secondary1_GeoAlbedoComp'] = -1
            Params['Secondary2_PlanetTemp'] = -1
            Params['Secondary3_MaxMES'] = -1
            Params['Secondary4_WeakSecondaryDepth'] = -1
            Params['Centroid1_RA_OOT'] = -1
            Params['Centroid2_deltaRA_OOT'] = -1
            Params['Centroid3_DEC_OOT'] = -1
            Params['Centroid4_deltaDEC_OOT'] = -1
            Params['Centroid5_CentroidOffset'] = -1

    else:
        data = pd.read_csv('./data/KTCE.csv', skiprows=161)
        specific = data[data['kepid'] == Params['ID']]
        if (len(specific) > 1):
            data = specific.iloc[0]
        else:
            data = specific
        Params['DV1_OpticalGhostCore'] = data['tce_cap_stat']
        Params['DV2_OpticalGhostHalo'] = data['tce_hap_stat']
        Params['DV3_BootstrapFalseAlarmProb'] = data['boot_fap']
        Params['DV4_RollingBandHist'] = data['tce_rb_tcount0']
        Params['DV5_OrbitalPeriod'] = data['tce_period']
        Params['DV6_PlanetRadius'] = data['tce_ror']
        Params['Secondary1_GeoAlbedoComp'] = data['tce_albedo_stat']
        Params['Secondary2_PlanetTemp'] = data['tce_ptemp']
        Params['Secondary3_MaxMES'] = data['tce_maxmes']
        Params['Secondary4_WeakSecondaryDepth'] = data['wst_depth']
        Params['Centroid1_RA_OOT'] = data['tce_fwm_sra']
        Params['Centroid2_deltaRA_OOT'] = data['tce_fwm_srao']
        Params['Centroid3_DEC_OOT'] = data['tce_fwm_sdec']
        Params['Centroid4_deltaDEC_OOT'] = data['tce_fwm_sdeco']
        Params['Centroid5_CentroidOffset'] = data['tce_fwm_stat']
    return Params


def TransitView(FoldedLightKurveObj, Duration, Centroid=False, featurename='TransitView', flux_str='WhitenedFlux',
                ID=None, mission=None, n_bins=31, Plot=False):
    """
    Generates Transit View Features
    """
    tv_durr = 5 * Duration / 24  # Duration is in hours #see pg 14 ExoMiner Algo2
    T_window = np.abs(tv_durr / 2)
    bin_width = Duration / 24 * .16
    data = FoldedLightKurveObj.bin(time_bin_size=bin_width, time_bin_start=-T_window, n_bins=n_bins)
    if (Plot):
        data.scatter()
        plt.savefig(mission + '_' + str(ID) + '_' + featurename + '.png')
        plt.close()
    if (Centroid == False):
        flux = data[flux_str].value.data
        phase = data['time'].value.data
    else:
        flux = data[flux_str].value
        phase = data['time'].value
    return {'phase': phase, 'flux': flux}


# takes pixels from observation and maps it to celestial coordinates
# takes a long time
def KeplerCentroidMotionFeatures(ID_Observation):
    """
    Computes centroid motion features for Kepler
    """
    for i in range(0, len(ID_Observation)):
        rdp = raDec2Pix.raDec2PixClass("ExtraPackages/raDec2PixDir")
        ID_Observation[i] = ID_Observation[i].remove_nans()
        rows = np.array(ID_Observation[i]['centroid_row'].value)
        cols = np.array(ID_Observation[i]['centroid_col'].value)
        BKJD = np.array(ID_Observation[i]['time'].value)
        module = [ID_Observation[i].meta['MODULE']] * len(rows)
        output = [ID_Observation[i].meta['OUTPUT']] * len(rows)
        JD = BKJD + 2454833.0  # convert to Julian Date
        mjd = JD - 2400000.5  # convert to modified julian date
        ra = []
        dec = []
        for j in range(0, len(rows)):
            # pix_2_ra_dec ported over from MATLAB
            ra_t, dec_t = rdp.pix_2_ra_dec(module[j], output[j], rows[j], cols[j], mjd[j])
            ra.append(ra_t)
            dec.append(dec_t)
        temp = a.coordinates.SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
        metrics = a.table.Table([temp.ra, temp.dec], names=['RA', 'DEC'])
        # adding to the lightcurve data
        ID_Observation[i].add_columns([metrics['RA']])
        ID_Observation[i].add_columns([metrics['DEC']])
    return ID_Observation


# takes pixels from observation and maps it to celestial coordinates
def TESSCentroidMotionFeatures(ID_Observation):
    """
    Computes centroid motion features for TESS
    """
    for i in range(0, len(ID_Observation)):

        ID_Observation[i] = ID_Observation[i].remove_nans()
        try:
            rows = np.array(ID_Observation[i]['centroid_row'].value)
            cols = np.array(ID_Observation[i]['centroid_col'].value)
        except:
            try:
                rows = np.array(ID_Observation[i]['sap_x'].value)
                cols = np.array(ID_Observation[i]['sap_y'].value)
            except:
                rows = np.array(ID_Observation[i]['xic'].value)
                cols = np.array(ID_Observation[i]['yic'].value)
        # BKJD    = np.array(ID_Observation[i]['time'].value)
        sector = [ID_Observation[i].meta['SECTOR']] * len(rows)
        camera = [ID_Observation[i].meta['CAMERA']] * len(rows)
        ccd = [ID_Observation[i].meta['CCD']] * len(rows)
        ra = []
        dec = []
        for j in range(0, len(rows)):
            ra_t, dec_t, _ = tess.tess_stars2px_reverse_function_entry(sector[j], camera[j], ccd[j], cols[j], rows[j])
            ra.append(ra_t)
            dec.append(dec_t)
        temp = a.coordinates.SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
        metrics = a.table.Table([temp.ra, temp.dec], names=['RA', 'DEC'])
        # adding to the lightcurve data
        ID_Observation[i].add_columns([metrics['RA']])
        ID_Observation[i].add_columns([metrics['DEC']])
    return ID_Observation


def CentroidMotionFeatures(LightkurveCollection, mission):
    """
    Helper function to push Lightkurve collection to proper centroid motion computation
    """
    if (mission == 'Kepler'):
        LightkurveCollection = KeplerCentroidMotionFeatures(LightkurveCollection)
        return LightkurveCollection
    elif (mission == 'TESS'):
        LightkurveCollection = TESSCentroidMotionFeatures(LightkurveCollection)
        return LightkurveCollection
    else:
        print('Please Choose valid Mission. Recall K2 is not valid.')
        return None


def GatherParams(DataObject, ID):
    """
    Gathers Scalar params for specific mission
    """
    if (DataObject.mission == 'Kepler'):
        Params = {
            'Mission': DataObject.mission,
            'ID': ID,
            'Disposition': DataObject.IdData['WrapperDispo'][DataObject.IdData['WrapperId'] == ID].values[0],
            'Period[Days]': DataObject.IdData['koi_period'][DataObject.IdData['WrapperId'] == ID].values[0],
            'StartTime[BKJD]': DataObject.IdData['koi_time0bk'][DataObject.IdData['WrapperId'] == ID].values[0],
            'FirstEpoch': DataObject.IdData['koi_time0bk'][DataObject.IdData['WrapperId'] == ID].values[0],
            'Duration[hrs]': DataObject.IdData['koi_duration'][DataObject.IdData['WrapperId'] == ID].values[0],
            'TransitDepth[ppm]': DataObject.IdData['koi_depth'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV1_TEFF': DataObject.IdData['koi_steff'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV2_StellarRadius': DataObject.IdData['koi_srad'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV3_StellarMass': DataObject.IdData['koi_smass'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV4_StellarDensity': DataObject.IdData['koi_srho'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV5_SurfaceGravity': DataObject.IdData['koi_slogg'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV6_Metalicity': DataObject.IdData['koi_smet'][DataObject.IdData['WrapperId'] == ID].values[0]
        }

    elif (DataObject.mission == 'TESS'):

        Params = {
            'Mission': DataObject.mission,
            'ID': ID,
            'Disposition': DataObject.IdData['WrapperDispo'][DataObject.IdData['WrapperId'] == ID].values[0],
            'Period[Days]': DataObject.IdData['pl_orbper'][DataObject.IdData['WrapperId'] == ID].values[0],
            'StartTime[BJD]': DataObject.IdData['pl_tranmid'][DataObject.IdData['WrapperId'] == ID].values[0],
            'StartTime[BTJD]': DataObject.IdData['pl_tranmid'][DataObject.IdData['WrapperId'] == ID].values[
                                   0] - 2457000.0,
            'FirstEpoch': DataObject.IdData['pl_tranmid'][DataObject.IdData['WrapperId'] == ID].values[0] - 2457000.0,
            # Planet Transit Midpoint Value [BJD]
            'Duration[hrs]': DataObject.IdData['pl_trandurh'][DataObject.IdData['WrapperId'] == ID].values[0],
            'TransitDepth[ppm]': DataObject.IdData['pl_trandep'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV1_TEFF': DataObject.IdData['st_teff'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV2_StellarRadius': DataObject.IdData['st_rad'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV3_StellarMass': 0,
            'SV4_StellarDensity': 0,
            'SV5_SurfaceGravity': DataObject.IdData['st_logg'][DataObject.IdData['WrapperId'] == ID].values[0],
            'SV6_Metalicity': 0

        }
    Params = OpenSuplamental(Params)
    return Params


def MaxObservtionsPerID(LightkurveCollection, max_val=3):
    """ Fucntion To limit number of observations used per ID"""
    if (len(LightkurveCollection) > max_val):
        NewCollection = LightkurveCollection[0:max_val]
    else:
        NewCollection = LightkurveCollection
    return NewCollection


def AverageOutOftransitCentroidPosition(FoldedLightKurveObj, Duration):
    """
    Computes AVG(C_OOT alpha)  AVG(C_OOT dec)
    """
    out_time = np.abs(3 * Duration / 24)
    out_time = out_time / 2
    out_time_low = -out_time
    out_time_high = out_time
    C_RA = FoldedLightKurveObj['RA']
    C_DEC = FoldedLightKurveObj['DEC']
    cond = np.logical_or(FoldedLightKurveObj.time.value < out_time_low, FoldedLightKurveObj.time.value > out_time_high)
    C_RA_out = np.mean(C_RA[cond])
    C_DEC_out = np.mean(C_DEC[cond])
    return (C_RA_out, C_DEC_out)


def RandGenForTest(DataObject):
    """
    Chooses a random ID for testing
    """
    l = len(DataObject.IdData['WrapperId'])
    return np.random.randint(low=0, high=l)


def Features(mish,DataObject, max_number_of_obs=3, Chosen_ID=None, Plot=False):
    """
    This funciton preforms data proccessing for feature generation
    """
    try:
        if (Chosen_ID == None): # This Block is used for testing with random IDs
            idx = RandGenForTest(DataObject)
            ID = DataObject.IdData['WrapperId'][idx]
        else:
            ID = Chosen_ID #choose spe
        Params = GatherParams(DataObject, ID)
        print(mish + f'ID {ID} '+ ' Planet Disposition ', Params['Disposition'])
        ID_Observation = DataObject.GatherIdLcData(id=ID, kind='All') # Gath
        ID_Observation = MaxObservtionsPerID(ID_Observation, max_val=max_number_of_obs)
        ID_Observation = CentroidMotionFeatures(ID_Observation, mission=mish)
        ID_Observation = StichObservations(ID_Observation, mission=mish)
        ID_Observation = Algo1Flux(ID_Observation, Params)
        ####
        ID_ObservationSecondary = Algo1FluxSecondary(ID_Observation, Params)
        ID_ObservationSecondary = ID_ObservationSecondary.select_flux('WhitenedFlux')
        ###
        ID_Observation = Algo1Centroid(ID_Observation, Params)
        ID_Observation = ID_Observation.select_flux('WhitenedFlux')
        ###
        ID_Observation_Folded = FoldLightKurveData(ID_Observation, Period=Params['Period[Days]'],
                                                   EpochStartTime=Params['FirstEpoch'])
        ID_Observation2_Folded = FoldLightKurveData(ID_ObservationSecondary, Period=Params['Period[Days]'],
                                                    EpochStartTime=Params['FirstEpoch'])

        Data = GenFeatures(FoldedLightKurveObj=ID_Observation_Folded, SecondaryLightKurveObj=ID_Observation2_Folded,
                           Parameters=Params, Plot=Plot)
        return Data
    except:
        print(mish ,f'{ID} failed remove this ID')
        BlacklistID(mish, ID) # remove bad ID
        return None # Return None


def BlacklistID(mission,ID):
    try:
        BadIds = np.load(mission + 'BlacklistedIDs.npy') #load list of mission specific bad Ids
        BadIds = np.append(BadIds, ID)
    except:
        BadIds=np.array([ID])
    np.save(mission + 'BlacklistedIDs.npy',BadIds) #Save New list of Bad Ids
    return None



def GrabFeatures(ID,DataObject, mish, Plot):
    """
    Helper Funciton that points to the Features funciton
    """
    Data = Features(mish=mish,DataObject=DataObject, max_number_of_obs=2, Chosen_ID=ID, Plot=Plot)
    return Data


def GenScalarFeatures(Parameters, Plot=False):
    '''Gather scalar parameter arrays for a single object ID'''
    TransitDepth = Parameters['TransitDepth[ppm]']

    StellarParamsScalars = np.array([Parameters['SV1_TEFF'],
                                     Parameters['SV2_StellarRadius'],
                                     Parameters['SV3_StellarMass'],
                                     Parameters['SV4_StellarDensity'],
                                     Parameters['SV5_SurfaceGravity'],
                                     Parameters['SV6_Metalicity']])
    StellarParamsScalars = np.nan_to_num(StellarParamsScalars, nan=-1)

    CentroidScalars = np.array([Parameters['Centroid1_RA_OOT'],
                                Parameters['Centroid2_deltaRA_OOT'],
                                Parameters['Centroid3_DEC_OOT'],
                                Parameters['Centroid4_deltaDEC_OOT'],
                                Parameters['Centroid5_CentroidOffset']])
    CentroidScalars = np.nan_to_num(CentroidScalars, nan=-1)
    SecondaryScalars = np.array([Parameters['Secondary1_GeoAlbedoComp'],
                                 Parameters['Secondary2_PlanetTemp'],
                                 Parameters['Secondary3_MaxMES'],
                                 Parameters['Secondary4_WeakSecondaryDepth']])

    SecondaryScalars = np.nan_to_num(SecondaryScalars, nan=-1)
    DVDiagnosticScalars = np.array([Parameters['DV1_OpticalGhostCore'],
                                    Parameters['DV2_OpticalGhostHalo'],
                                    np.log(Parameters['DV3_BootstrapFalseAlarmProb'] + 0.00000001),
                                    # log before normalization
                                    Parameters['DV4_RollingBandHist'],
                                    Parameters['DV5_OrbitalPeriod'],
                                    Parameters['DV6_PlanetRadius']])
    DVDiagnosticScalars = np.nan_to_num(DVDiagnosticScalars, nan=-1)

    Data = {
        'TransitDepth': TransitDepth,
        'CentroidScalars': CentroidScalars,
        'SecondaryScalars': SecondaryScalars,
        'StellarParamsScalars': StellarParamsScalars,
        'DVDiagnosticScalars': DVDiagnosticScalars
    }
    return Data


def normalize_scalars(DataObject):
    n = len(DataObject.IdData['WrapperId'])
    transit_depth = np.zeros((n, 1))
    centroid_scalars = np.zeros((n, 5))
    secondary_scalars = np.zeros((n, 4))
    stellar_params = np.zeros((n, 6))
    dv_diag_scalars = np.zeros((n, 6))
    scalar_list = [transit_depth, centroid_scalars, secondary_scalars, stellar_params, dv_diag_scalars]
    scalar_names = ['TransitDepth', 'CentroidScalars', 'SecondaryScalars', 'StellarParamsScalars',
                    'DVDiagnosticScalars']

    # create arrays for each scalar param (for all IDs)
    t_start = t.time()
    for num, id in enumerate(DataObject.IdData['WrapperId']):

        if(num%100 == 0 ):
            t_stop = t.time()
            print(num,' took '+str(t_start-t_stop)+'s')
            t_start = t.time()


        Params = GatherParams(DataObject, id)

        scalar_data = GenScalarFeatures(Parameters=Params)

        # print(scalar_data)
        try:
            transit_depth[num, :] = scalar_data['TransitDepth'].reshape((1))
        except:
            transit_depth[num, :] = np.array([-1]).reshape((1))
        try:
            centroid_scalars[num, :] = scalar_data['CentroidScalars'].reshape((5))
        except:
            centroid_scalars[num, :] = np.array([-1, -1, -1, -1, -1]).reshape((5))
        try:
            secondary_scalars[num, :] = scalar_data['SecondaryScalars'].reshape((4))
        except:
            secondary_scalars[num, :] = np.array([-1, -1, -1, -1]).reshape((4))
        try:
            stellar_params[num, :] = scalar_data['StellarParamsScalars'].reshape((6))
        except:
            stellar_params[num, :] = np.array([-1, -1, -1, -1, -1, -1]).reshape((6))
        try:
            dv_diag_scalars[num, :] = scalar_data['DVDiagnosticScalars'].reshape((6))
        except:
            dv_diag_scalars[num, :] = np.array([-1, -1, -1, -1, -1, -1]).reshape((6))


    scalar_medians = {}
    scalar_stdevs = {}
    output_data = {}

    # normalize values (algo 3)
    for X, name in zip(scalar_list, scalar_names):
        med_X = np.median(X, axis=0)
        std_X = np.std(X, axis=0)
        X_temp = (X - med_X) / std_X
        X_temp_clip = np.clip(X_temp, -20 * std_X, 20 * std_X)
        X = X_temp_clip

        # capture each normalized array
        output_data[name] = X_temp_clip

        scalar_medians[name] = med_X
        scalar_stdevs[name] = std_X

    normalized_scalars = {}

    # reassign normalized values in Data dictionary
    for num, id in enumerate(DataObject.IdData['WrapperId']):
        normalized_scalars[id] = {'TransitDepth': output_data['TransitDepth'][num, :],
                                  'CentroidScalars': output_data['CentroidScalars'][num, :],
                                  'SecondaryScalars': output_data['SecondaryScalars'][num, :],
                                  'StellarParamsScalars': output_data['StellarParamsScalars'][num, :],
                                  'DVDiagnosticScalars': output_data['DVDiagnosticScalars'][num, :]
                                  }
    SaveScalarsDict(normalized_scalars,mission =DataObject.mission )  #Save as numpy array so dont have to recraete

    return normalized_scalars, scalar_medians, scalar_stdevs

def SaveScalarsDict(normalized_scalars,mission):
    """
    Save Scalars so we dont have to recompute each time
    """
    np.save(mission + 'Scalars.npy',normalized_scalars )
    return None

def calc_and_store_features(DataObject, mission, restrict_IDs=False, max_ID_num=5):
    '''
    Note: when restrict_IDs is true, the total number of IDs processed will be limited to max_ID_num 
    '''
    try:
        normalized_scalars = np.load(DataObject.mission + 'Scalars.npy',allow_pickle=True).item() # load npy array
    except:
        normalized_scalars, _, _ = normalize_scalars(DataObject)

    final_features = {}
    batch =0

    # reassign normalized values in Data dictionary
    for num, id in enumerate(DataObject.IdData['WrapperId']):

        if ((restrict_IDs) and (num > max_ID_num)):
            break

        Features = GrabFeatures(id,DataObject, mish=mission, Plot=False)  # gets features for ID
        if(Features==None):
            continue # if there is an error downloading this specific index skip it
        scalar_names = ['TransitDepth', 'CentroidScalars', 'SecondaryScalars', 'StellarParamsScalars',
                        'DVDiagnosticScalars']

        for name in scalar_names:
            Features[name] = normalized_scalars[id][name]

        final_features[id] = Features
        if(num%100 ==0 and num>0):
            SaveFeatures(mission, final_features,batch)
            final_features = {}
            batch+=1
    batch+=1
    SaveFeatures(mission, final_features,batch)

def SaveFeatures(mission,features,batch):

    np.save(mission + f'Features_{batch}.npy', features)
    print('Succesfully saved Features')


def Benchmark(Func, mission, ID=None):
    start_time = t.time()
    if (ID == None):
        Func(mission)
    else:
        Func(mish=mission, Chosen_ID=ID)
    end_time = t.time()
    print(mission + f' test took {end_time - start_time} seconds')

def interp_nans(y, x=None):
    if x is None:
        x = np.arange(len(y))
    nans = np.isnan(y)
    if(any(nans)):
        return y
    else:
        interpolator = interpolate.interp1d(
            x[~nans],
            y[~nans],
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
    y[nans] = interpolator(x)[nans]
    return y
def GetRawData(mission,DataObject,Nbins):
    ID = DataObject.IdData['WrapperId']
    data = {}
    count = 0
    i=0
    for id in ID:
        try:
            ID_Observation = DataObject.GatherIdLcData(id=id, kind='All')
            ID_Observation = MaxObservtionsPerID(ID_Observation, max_val=2)
            ID_Observation = StichObservations(ID_Observation, mission=mission)
            try:
                temp = ID_Observation['centroid_row']
                flag = True
            except:
                try:
                    temp =ID_Observation['xic']
                    flag = True
                except:
                    try:
                        temp = ID_Observation['sap_x']
                        flag = True
                    except:
                        flag = False
            if(flag):
                Params = GatherParams(DataObject, id)
                ID_Observation_Folded = FoldLightKurveData(ID_Observation, Period=Params['Period[Days]'],
                                                           EpochStartTime=Params['FirstEpoch'])
                min_t =ID_Observation_Folded['time'].min().value
                max_t = ID_Observation_Folded['time'].max().value
                bin_w = (max_t +abs(min_t))/6000
                binned = ID_Observation_Folded.bin(time_bin_size=bin_w , time_bin_start=min_t ,n_bins=Nbins)
                try:

                    flux = interp_nans(binned['flux'].value)
                    centroid_row = interp_nans(binned['centroid_row'].value)
                    centroid_col = interp_nans(binned['centroid_col'].value)

                except:
                    try:
                        flux = interp_nans(binned['flux'].value)
                        centroid_row = interp_nans(binned['xic'].value)
                        centroid_col = interp_nans(binned['yic'].value)
                    except:
                        flux = interp_nans(binned['flux'].value)
                        centroid_row = interp_nans(binned['sap_x'].value)
                        centroid_col = interp_nans(binned['sap_y'].value)

                data[id] = {'Flux': flux,
                            'Centroid Row': centroid_row,
                            'Centroid Col': centroid_col}
                count +=1
                if(count%5 == 0 ):
                    i+=1
                    np.save('RawData/'+mission+f'/dataset_{i}.npy',data)
                    data= {}
            else:
                print(mission + f' ID {id} was not included')
        except:
            print(mission+f' ID {id} was not included')
    return None




if __name__ == '__main__':
    # Benchmark(Features, 'Kepler') #ID input as int(10419211)
    #Benchmark(Features, 'TESS')  # ID input as str('TIC 102542376')
    pass
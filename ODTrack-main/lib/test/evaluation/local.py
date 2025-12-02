from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data_F/jinglin/GRcode/ODTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/data_F/datasets/got10k/got_10k_data/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data_F/jinglin/GRcode/ODTrack-main/data/itb'
    settings.lasot_extension_subset_path = '/data_F/datasets/LaSOT_ext/LaSOT_extension_subset/'
    settings.lasot_lmdb_path = '/data_F/jinglin/GRcode/ODTrack-main/data/lasot_lmdb'
    settings.lasot_path = '/data_F/datasets/LaSOTBenchmark/'
    settings.network_path = '/data_F/jinglin/GRcode/ODTrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data_F/jinglin/GRcode/ODTrack-main/data/nfs'
    settings.otb_path = '/data_F/datasets/dataset_OTB/'
    settings.prj_dir = '/data_F/jinglin/GRcode/ODTrack-main'
    settings.result_plot_path = '/data_F/jinglin/GRcode/ODTrack-main/output/test/result_plots/'
    settings.results_path = '/data_F/jinglin/GRcode/ODTrack-main/output/test/tracking_results/'    # Where to store tracking results
    settings.save_dir = '/data_F/jinglin/GRcode/ODTrack-main/output'
    settings.segmentation_path = '/data_F/jinglin/GRcode/ODTrack-main/output/test/segmentation_results'
    settings.tc128_path = '/data_F/datasets/Temple-color-128/'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data_F/jinglin/GRcode/ODTrack-main/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data_F/datasets/TrackingNet/'
    settings.uav_path = '/data_F/datasets/UAV123/'
    settings.vot18_path = '/data_F/jinglin/GRcode/ODTrack-main/data/vot2018'
    settings.vot22_path = '/data_F/jinglin/GRcode/ODTrack-main/data/vot2022'
    settings.vot_path = '/data_F/jinglin/GRcode/ODTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


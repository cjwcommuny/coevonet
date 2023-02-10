from gcn_ranker.taobao_runner.taobao_frames_runner import TaobaoFramesRunner
from gcn_ranker.taobao_runner.taobao_random_runner import TaobaoRandomRunner
from gcn_ranker.taobao_runner.taobao_video2gif_runner import TaobaoVideo2gifRunner
from gcn_ranker.taobao_runner.taobao_window_cls_runner import TabaoWindowClsRunner
from gcn_ranker.taobao_runner.taobao_window_regression_runner import TaobaoWindowRegressionRunner
from gcn_ranker.youtube_runner.youtube_gcn_naive_runner import YoutubeGcnNaiveRunner
from gcn_ranker.youtube_runner.youtube_multimodal_runner import YoutubeMultiModalRunner
from gcn_ranker.youtube_runner.youtube_ranknet_runner import YoutubeRanknetRunner
from gcn_ranker.youtube_runner.youtube_scorer_selector_runner import YoutubeScorerSelectorRunner
from gcn_ranker.taobao_runner.taobao_sqan_runner import TaobaoSqanRunner
from gcn_ranker.taobao_runner.taobao_pairwise_runner import TaobaoPairwiseRunner

RUNNING_TYPES = {
    'taobao_sqan_runner': TaobaoSqanRunner,
    'youtube_highlight_ranknet': YoutubeRanknetRunner,
    'youtube_highlight_scorer_selector': YoutubeScorerSelectorRunner,
    'youtube_highlight_gcn': YoutubeGcnNaiveRunner,
    'youtube_multi_modal': YoutubeMultiModalRunner,
    'taobao_window_cls_runner': TabaoWindowClsRunner,
    'taobao_window_regression_runner': TaobaoWindowRegressionRunner,
    'taobao_frames_runner': TaobaoFramesRunner,
    'taobao_random_runner': TaobaoRandomRunner,
    'taobao_video2gif_runner': TaobaoVideo2gifRunner,
    'taobao_pairwise_runner': TaobaoPairwiseRunner
}
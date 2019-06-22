import numpy as np


class TopicCorelevanceBuilder:
    """ Build topic-to-topic similarity matrix using text-to-topic relevance matrix """

    def __init__(self, score_df, relevance_thresh):
        relevants = (score_df > relevance_thresh).sum(axis=1)
        max_relevant = relevants.max()
        relevance_weights = np.sqrt(relevants/max_relevant)
        scores_wt = score_df.values * relevance_weights[:, None]
        corelevance = np.dot(scores_wt.T, scores_wt)

        self.score_df = score_df
        self.relevance_thresh = relevance_thresh
        self.relevants = relevants
        self.relevance_weights = relevance_weights
        self.scores_wt = scores_wt
        self.corelevance = corelevance


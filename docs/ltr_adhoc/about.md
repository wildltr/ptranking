
## About LTR_Adhoc
By **LTR_Adhoc**, we refer to the traditional learning-to-rank methods based on the Empirical Risk Minimization Framework, which is detailed in [ptranking_empirical_risk_minimization.ipynb](https://github.com/ptranking/ptranking.github.io/raw/master/tutorial/).

Major learning-to-rank approaches can be classified into three categories: **pointwise**, **pairwise**, and **listwise**. The key distinctions are the underlying hypotheses, loss functions, the input and output spaces.

The typical pointwise approaches include regression-based [1], classification-based [2], and ordinal regression-based algorithms [3, 4]. The loss functions of these algorithms is defined on the basis of each individual document.

The pairwise approaches care about the relative order between two documents. The goal of learning is to maximize the number of correctly ordered document pairs. The assumption is that the optimal ranking of documents can be achieved if all the document pairs are correctly ordered. Towards this end, many representative methods have been proposed [5,6,7,8,9].

The listwise approaches take all the documents associated with the same query in the training data as the input. In particular, there are two types of loss functions when performing listwise learning. For the first type, the loss function is related to a specific evaluation metric (e.g., nDCG and ERR). Due to the non-differentiability and non-decomposability of the commonly used metrics, the methods of this type either try to optimize the upper bounds as surrogate objective functions [10, 11, 12] or approximate the target metric using some smooth
functions [13, 14, 15]. However, there are still some open issues regarding the first type methods. On one hand, some adopted surrogate functions or approximated metrics are not convex, which makes it hard to optimize. On the other hand, the relationship between the surrogate function and the adopted metric has not been sufficiently investigated, which makes it unclear whether optimizing the surrogate functions can indeed optimize the target metric. For the second type, the loss function is not explicitly related to a specific evaluation metric. The loss function reflects the discrepancy between the predicted ranking and the ground-truth ranking. Example algorithms include []. Although no particular evaluation metrics are directly involved and optimized here, it is possible that the learned ranking function can achieve good performance in terms of evaluation metrics.

## References

- [1] David Cossock and Tong Zhang. 2006. Subset Ranking Using Regression. In
Proceedings of the 19th Annual Conference on Learning Theory. 605–619.
- [2] Ramesh Nallapati. 2004. Discriminative Models for Information Retrieval. In
Proceedings of the 27th SIGIR. 64–71.
- [3] Wei Chu and Zoubin Ghahramani. 2005. Gaussian Processes for Ordinal Regression.
Journal of Machine Learning Research 6 (2005), 1019–1041.
- [4] Wei Chu and S. Sathiya Keerthi. 2005. New Approaches to Support Vector Ordinal
Regression. In Proceedings of the 22nd ICML. 145–152.
- [5] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton,
and Greg Hullender. 2005. Learning to rank using gradient descent. In Proceedings
of the 22nd ICML. 89–96.
- [6] Yoav Freund, Raj Iyer, Robert E. Schapire, and Yoram Singer. 2003. An Efficient
Boosting Algorithm for Combining Preferences. Journal of Machine Learning
Research 4 (2003), 933–969.
- [7] Thorsten Joachims. 2002. Optimizing search engines using clickthrough data. In
Proceedings of the 8th KDD. 133–142.
- [8] Libin Shen and Aravind K. Joshi. 2005. Ranking and Reranking with Perceptron.
Machine Learning 60, 1-3 (2005), 73–96.
- [9] Fajie Yuan, Guibing Guo, Joemon Jose, Long Chen, Hai-Tao Yu, andWeinan Zhang.
2016. LambdaFM: Learning Optimal Ranking with Factorization Machines Using
Lambda Surrogates. In Proceedings of the 25th CIKM. 227–236.
- [10] Olivier Chapelle, Quoc Le, and Alex Smola. 2007. Large margin optimization of
ranking measures. In NIPS workshop on Machine Learning for Web Search.
- [11] Jun Xu and Hang Li. 2007. AdaRank: a boosting algorithm for information
retrieval. In Proceedings of the 30th SIGIR. 391–398.
- [12] Yisong Yue, Thomas Finley, Filip Radlinski, and Thorsten Joachims. 2007. A
Support Vector Method for Optimizing Average Precision. In Proceedings of the
30th SIGIR. 271–278.
- [13] John Guiver and Edward Snelson. 2008. Learning to Rank with SoftRank and
Gaussian Processes. In Proceedings of the 31st SIGIR. 259–266.
- [14] Tao Qin, Tie-Yan Liu, and Hang Li. 2010. A general approximation framework
for direct optimization of information retrieval measures. Journal of Information
Retrieval 13, 4 (2010), 375–397.
- [15] Michael Taylor, John Guiver, Stephen Robertson, and Tom Minka. 2008. SoftRank:
Optimizing Non-smooth Rank Metrics. In Proceedings of the 1st WSDM. 77–86.
- [16] Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006. Learning to Rank
with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
- [17] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to
Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th
ICML. 129–136.
- [18] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise
Approach to Learning to Rank: Theory and Algorithm. In Proceedings of the 25th
ICML. 1192–1199.

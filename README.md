# K-Means Clustering from Scratch
 Developing unsupervised learning algorithm K-Means from scratch and test performance compared to sklearn package
## 1 Introduction
### 1.1 What is K-Means Clustering?
K-Means clustering is a commonly used non-hierarchical unsupervised machine learning
algorithm that divides a dataset into K distinct clusters so that clusters are as homogeneous
as possible. The algorithm functions by minimizing the variance (maximizing similarity)
within each cluster while maximizing the variance (minimizing similarity) between different
clusters.
### 1.2 Project Motivation
K-Means clustering is a powerful and widely-used data mining tool that has lots of applications
in the real world. For example, it can be used in market segmentation to divide
customers into different groups based on their preferences, purchase history, and other factors.
It can also be used to classify images and detect patterns in text documents.
## 2 Optimization Problem and Formulation
### 2.1 K-means in Optimization Problem & Methodology
The optimization problem for K-means is to find the values of K centroids that minimize
the objective function (Equation 1). Suppose the dataset we have {x1, x2, ..., xn} observations
of a 2-dimensional variable x. We first randomly select k points to serve as the initialized
centroids {c1, c2, ..., ck}. Secondly, calculate the Euclidean distance (Equation 2) between
the observation xi with each centroid and assign the observation xi to the nearest centroid.
We will continue iterations until the algorithm converges and update the centroid positions
to minimize the objective function 1 (Gopalakrishnan, 2023) [3].
### 2.2 Mathematical Formulation
#### 2.2.1 Objectives
Our main goal is to minimize the sum of squared distances of each data point to its assigned
centroid of kth cluster ck:
![image](https://user-images.githubusercontent.com/78404450/223493865-f014f525-aee1-4c09-9ec4-52650c1a34e3.png)
![image](https://user-images.githubusercontent.com/78404450/223493962-234ed4e4-e641-42cd-9de7-0c3d1af558c7.png)

The objective function of K-Means is non-convex with respect to the cluster centers because
it involves a sum of squared distances, which is a non-convex function. We can state that KMeans
is a non-convex function that we cannot find the global optimal since the algorithm
may converge to a local minimum rather than the global minimum.

### 2.3 Methodology for Test Performance
To test the performance of the K-Means algorithm, we can use metrics such as the Within
Cluster Sum of Squares (WCSS) and the silhouette coefficient.

## 3 Experiment with datasets
### 3.1 The Iris Dataset
The iris dataset is a popular dataset used in machine learning for classification tasks. It
contains information about the physical characteristics of 150 iris flowers belonging to three
different species: Setosa, Versicolor, and Virginica. The dataset includes four features for
each flower: sepal length, sepal width, petal length, and petal width, all measured in centimeters.
Upon implementation by running K-means on the Iris dataset with K=3, we have the following
result in Figure
![image](https://user-images.githubusercontent.com/78404450/223494674-690f8d26-8948-4daa-97b1-4012f0039d12.png)

### 3.2 Customer Segmentation Dataset
Customer Segmentation is the process of dividing a market into distinct groups of customers
that share similar characteristics. By identifying and understanding customer needs through
segmentation, companies can develop unique products and services that outperform the
competition. In the context of a supermarket mall, customer data such as Customer ID,
age, gender, annual income, and spending score can be used to identify target customers and
inform marketing strategies. We aim to analyze a customer segmentation dataset and provide
insights into customer behavior, preferences, and needs.

Upon implementation by running K-means on the dataset with K=5, we have the following
result in Figure 6:
![image](https://user-images.githubusercontent.com/78404450/223495056-9f6a9c5b-c3e5-4700-87f5-0c5920f07309.png)
![image](https://user-images.githubusercontent.com/78404450/223495139-2a7506a2-ed52-48cd-a946-ac7971a1d8f6.png)

## 4 Performance Comparison
Examining Table 1, the performance results of our implementation on the Iris dataset demonstrate
that it is relatively worse compared to the Scikit-Learn package. While the WCSS is
similar, our implementation only has a Silhouette Coefficient of 0.245, whereas the Scikit-
Learn package displays a Silhouette Coefficient of 0.445, indicating that the package ensures
that the groups are distinct between groups but more homogeneous within the groups. The running times of the two methods are comparable (0.642 seconds vs. 0.515 seconds). 
#
![image](https://user-images.githubusercontent.com/78404450/223495375-27ed18fd-f6c5-49ed-bf66-b3f1a658cb98.png) 
#
As indicated by Table 2, when our technique is implemented on a larger dataset,
our implementation requires a dramatically longer running time, which is approximately 18
times slower compared to the Scikit-Learn package. Despite this, the performance of both
methods is similar, with almost the same WCSS and Silhouette Coefficients. When evaluating
the performance between the two datasets, the Customer Segmentation Dataset has an
exceptionally larger WCSS compared to the Iris Dataset. This is because the income factor
in the former has a larger unit, and thus WCSS provides a limited indication when comparing
across datasets. Moreover, the Silhouette Coefficients of both methods on the customer
segmentation dataset are higher compared to the Iris datasets. A likely explanation for the
better performance is that the Customer Segmentation dataset contains more observations
and thus can better train the model to achieve a better result.
In terms of running time, our implementation performs differently across datasets. It can
yield good efficiency when the dataset is small, yet require a long time to run when the dataset
is large. However, in comparison, the Scikit-Learn implementation appears unaffected by the
size of the dataset. This is likely due to its use of K-Means++ for the centroid initialization,
which seeks to spread out the initial centroids by assigning the first centroid randomly, then
select the rest of the centroids based on the maximum squared distance (Sharma, 2023) [7].
Our results demonstrate that this centroid initialization can significantly improve the speed
of the model compared to selecting the initialized centroid randomly.

## 5 Conclusions
For this project, we focused on introducing the concept of K-Means clustering, an effective
unsupervised machine learning algorithm that seeks to minimize the variance within each
cluster while maximizing the variance between different clusters. We discussed the application
of optimization to K-Means in order to obtain the desired clustering results, including
the utilization of the Euclidean distance to measure distance, and the Within Cluster Sum of
Squares (WCSS) and the Silhouette Coefficient to evaluate performance. We then provided
a demonstration by applying the procedures to two different datasets: the Iris dataset and
the Customers Segmentation Dataset. To compare the performance of our implementation,
we also used the Scikit-Learn package to conduct similar clustering processes on the two
datasets. We concluded that, for small datasets, the optimization techniques on K-Means
had a poorer performance, albeit with good efficiency; whereas for large datasets, it had
better performance, but required dramatically more resources to run.


# Recommendation Systems and Association Rule Learning

Recommendation systems are essential in suggesting content, products, and services to users. While they have been around since the 90s, their usage expanded significantly in data analytics since 2009.

## Overview

- **Content Personalization**: Users have varied interests within a vast content pool. Recommendation systems aid in providing personalized services by filtering this diverse content.
  
- **Applications**: Widely used in dating apps, e-commerce sites, social media channels, and more.

## Types of Recommendation Systems

1. **Simple Recommender Systems**: Offers general recommendations using business knowledge or basic sequencing techniques.
   
2. **Association Rule Learning**: Derives suggestions from learned rules through association analysis.
   
3. **Content-Based Filtering**: Recommends based on product similarities.
   
4. **Collaborative Filtering**: Provides recommendations based on user or product preferences. Divided into User-Based, Product-Based, and Model-Based methods.

## Simple Recommender Systems

These systems focus on popular, high-rated, or legendary products, disregarding user behavior or product features.

## Association Rule Learning

Utilizes rule-based machine learning, employing the Apriori Algorithm for association analysis.

### Apriori Algorithm Metrics

- **Support**: Measures the frequency of products X and Y being purchased together.
- **Confidence**: Probability of purchasing product Y when product X is purchased.
- **Lift**: Coefficient of increase in the probability of purchasing product Y when product X is purchased.
#### 1. Support

- **Definition**: Measures the frequency of items X and Y being purchased together.
- **Formula**: Support(X, Y) = Freq(X, Y) / Total Transaction

#### 2. Confidence

- **Definition**: Measures the likelihood of item Y being purchased when item X is purchased.
- **Formula**: Confidence(X, Y) = Freq(X, Y) / Freq(X)

#### 3. Lift

- **Definition**: Measures the increase in likelihood of item Y being purchased when item X is purchased.
- **Formula**: Lift = Support(X, Y) / (Support(X) * Support(Y))

#### 4. Leverage

- **Definition**: Similar to Lift but prioritizes higher support values.
- **Formula**: Leverage = Support(X, Y) - (Support(X) * Support(Y))

#### 5. Conviction

- **Definition**: Measures the likelihood of item X appearing without item Y for highly reliable rules.
- **Formula**: Conviction = (1 - Support(Y)) / (1 - Confidence(X, Y))


### How Apriori Works

1. Calculate the support value of each product.
2. Eliminate products below the support threshold.
3. Identify possible product pairs and their support values.
4. Iterate to refine possible pairs until a final table is achieved.


## Python Implementation - ARL

This project includes a Python implementation using the online_retail_II dataset, focusing on implementing Association Rule Learning with the Apriori Algorithm.


## Product Recommendation

The final step involves making product recommendations based on association rules and the implemented algorithm.



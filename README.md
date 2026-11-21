# Causal Impact of Delivery Delay on Customer Review Scores & Retention (Sharp RDD)

## 1. Research Question

**Does crossing the promised delivery deadline causally hurt customer review scores and future repeat purchase behavior in e-commerce?**

Using the **Olist Brazilian E-Commerce dataset** (96,470 delivered orders merged across 9 relational tables; 93,350 first-time orders), this project implements a **Sharp Regression Discontinuity Design (RDD)** around the promised delivery cutoff (`delay_days = 0`) to separate the causal effect of late delivery from confounding seller quality, product attributes, and seasonality.

## 2. Data Engineering & Relational Schema Integrity

Multi-table e-commerce logs are prone to `1-to-Many` join explosions that silently double-count orders. Before econometric estimation, the Python ETL pipeline enforces strict `1-to-1` order-level granularity:

* **Pre-Aggregation (`1-to-1` Enforcement):**
  * `order_payments`: Aggregated at the `order_id` level (`total_payment`, `max_installments`, `n_payment_methods`).
  * `order_items`: Aggregated at the `order_id` level (`total_price`, `total_freight`, `n_items`, `n_sellers`), retaining the highest-priced item's `seller_id` and `product_id` as primary fixed-effect attributes.
  * `order_reviews`: Deduplicated at the `order_id` level by keeping the most recent review timestamp.
  * Verified strict `1-to-1` row-count preservation (`N = 96,470`) via programmatic `assert` checks.
* **Running Variable Construction (`delay_days`):**
  * Truncated both `order_delivered_customer_date` and `order_estimated_delivery_date` to the **date level** (`.dt.floor("D")`) before differencing so that orders arriving on the afternoon of Day 0 are accurately classified as on-time (`delay_days = 0`), placing the treatment cutoff strictly at `delay_days > 0`.
* **Retention & Right-Censoring Adjustment:**
  * Tracked chronological repeat purchases via `customer_unique_id`.
  * Restricted main estimation to **first-time orders** (`is_first_order == 1`) to eliminate within-customer serial correlation, and excluded orders lacking a full 90-day post-purchase observation window (`valid_90d_window == 1`) for the 90-day retention outcome.

### Descriptive Overview

Across all 96,470 delivered orders, **6.77%** arrive late (`delay_days > 0`). In a naive comparison, delayed orders score **2.02 stars lower** on average (`2.271` vs. `4.290` stars). However, this raw gap conflates severe multi-week delays with unobserved seller and product differences—motivating the local Sharp RDD design. Baseline repeat purchase rates among first-time buyers (with a valid 90-day window) are low overall (**3.52%** any-time repeat; **2.30%** within 90 days for on-time orders).

## 3. Empirical Strategy & Identification Validity

We estimate local linear regressions within a symmetric bandwidth of `delay_days` in `[-7, 7]` around the cutoff:

$$Y_{ist} = \alpha + \tau \cdot \mathbb{I}(DelayDays_i > 0) + \beta_1 DelayDays_i + \beta_2 \mathbb{I}(DelayDays_i > 0) \times DelayDays_i + \mu_s + \lambda_c + \delta_t + \varepsilon_{ist}$$

where $\mu_s$, $\lambda_c$, and $\delta_t$ denote `main_seller_id`, `product_category_name_english`, and `purchase_ym` fixed effects, respectively, with standard errors clustered by `main_seller_id`.

* **No Manipulation at Cutoff:** Daily order counts across `[-7, 7]` decline smoothly across the threshold (`1,462` at Day -1, `1,292` at Day 0, `825` at Day +1), showing no abnormal bunching right before the deadline.
* **Covariate Balance (Placebo RDD Tests, `BW = [-7, 7]`):** Pre-determined order characteristics exhibit no statistically significant discontinuity at the cutoff:
  * Order Price (`total_price`): `-1.70` (`SE = 5.43`)
  * Freight Value (`total_freight`): `+0.41` (`SE = 0.66`)
  * Product Weight (`product_weight_g`): `+55.73g` (`SE = 103.40`)
  * *(Note: Because `total_price` displays a slight post-cutoff slope of `+3.05*` (`SE = 1.30`), Section 5 explicitly verifies stability when adding all three covariates directly as regression controls.)*

## 4. Main Results

### Regression Summary (`fixest`, First-Time Buyers, Bandwidth = `[-7, 7]`)

| Dependent Variable | (1) Review Score (No FE) | (2) Review Score (Full FE) | (3) Repeat Order (Any) | (4) Repeat Order (90d) |
| :--- | :---: | :---: | :---: | :---: |
| **`is_delayed` (Cutoff Jump $\tau$)** | **$-0.1737^{**}$** $(0.0537)$ | **$-0.1102^{*}$** $(0.0555)$ | $0.0019$ $(0.0061)$ | $0.0035$ $(0.0065)$ |
| **`delay_days` (Pre-cutoff Slope)** | $-0.0274^{***}$ $(0.0043)$ | $-0.0308^{***}$ $(0.0045)$ | $-0.0003$ $(0.0005)$ | $-0.0006$ $(0.0005)$ |
| **`is_delayed × delay_days` (Slope Kink)** | **$-0.2987^{***}$** $(0.0118)$ | **$-0.2868^{***}$** $(0.0121)$ | $-0.0002$ $(0.0015)$ | $0.0007$ $(0.0015)$ |
| Seller, Category & Year-Month FE | No | Yes | Yes | Yes |
| Clustered SE (`main_seller_id`) | Yes | Yes | Yes | Yes |
| Observations | 21,438 | 20,849 | 21,032 | 15,657 |
| $R^2$ | 0.1955 | 0.2866 | 0.0835 | 0.0868 |

*Note: Standard errors clustered by `main_seller_id` in parentheses. $^\cdot p < 0.10$, $^* p < 0.05$, $^{**} p < 0.01$, $^{***} p < 0.001$.*

### Visual Evidence



## 5. Robustness Checks: Bandwidth Sensitivity, Controls & Kernel Weighting

To test whether the review score effect is sensitive to bandwidth selection, pre-determined covariates, or local weighting near Day 0, we re-estimate the Full FE specification across five variants:

| Specification | (1) BW `[-5, 5]` | (2) BW `[-7, 7]` (Main) | (3) BW `[-10, 10]` | (4) BW `[-7, 7]` + Ctrls | (5) Triangular Kernel `[-7, 7]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`is_delayed` (Cutoff Jump)** | $0.0399$ $(0.0680)$ | **$-0.1102^{*}$** $(0.0555)$ | **$-0.3186^{***}$** $(0.0470)$ | $-0.1085^\cdot$ $(0.0554)$ | $-0.0141$ $(0.0596)$ |
| **`delay_days` (Pre-cutoff Slope)** | $-0.0224^{**}$ $(0.0081)$ | $-0.0308^{***}$ $(0.0045)$ | $-0.0280^{***}$ $(0.0028)$ | $-0.0297^{***}$ $(0.0045)$ | $-0.0289^{***}$ $(0.0050)$ |
| **`is_delayed × delay_days` (Slope Kink)** | **$-0.3733^{***}$** $(0.0223)$ | **$-0.2868^{***}$** $(0.0121)$ | **$-0.2237^{***}$** $(0.0079)$ | **$-0.2854^{***}$** $(0.0121)$ | **$-0.3289^{***}$** $(0.0147)$ |
| Pre-determined Controls | No | No | No | Yes | No |
| Seller, Category & Year-Month FE | Yes | Yes | Yes | Yes | Yes |
| Observations | 11,868 | 20,849 | 35,220 | 20,847 | 20,849 |
| $R^2$ | 0.2737 | 0.2866 | 0.2694 | 0.2913 | 0.2629 |

* **Controls Stability:** Adding `total_price`, `total_freight`, and `product_weight_g` leaves both the jump (`-0.1085` vs. `-0.1102`) and slope kink (`-0.2854***` vs. `-0.2868***`) virtually unchanged.
* **Discrete Running Variable & Non-Parametric `rdrobust`:** Estimating `rdrobust` (`c = 0.5`, `h = 7`, `masspoints = "adjust"`) yields a noisy, insignificant level jump (`-0.213`, robust `p = 0.903`, 95% CI `[-0.163, 0.144]`). Combined with the shrinking jump under narrower bandwidths (`[-5, 5]`) and Triangular Kernel weighting (`-0.0141`), this indicates that the apparent intercept jump at wider windows (`[-10, 10]`) is partly driven by **post-cutoff concavity** as ratings approach the 1-star floor.
* **Core Robust Signal (Slope Kink):** Across every bandwidth, kernel, and control specification, the **post-cutoff slope change (`is_delayed × delay_days`)** remains highly significant ($p < 0.001$), strengthening to **`-0.33` to `-0.37` stars/day** closest to the cutoff.

## 6. Key Takeaways

1. **Strong Asymmetric Penalty & Slope Kink in Ratings:** Delivering early yields minimal marginal rating gains (~`0.03` stars/day), whereas crossing the promised delivery deadline triggers a sharp **slope kink**—increasing the marginal penalty more than tenfold to roughly **`0.32`–`0.39` stars per additional day of delay** near the cutoff.
2. **Null Effect on Platform-Level Retention:** Delivery delays show **no statistically significant causal impact** on future platform-level repeat purchases (any-time or within 90 days). In a multi-seller marketplace like Olist, dissatisfied buyers penalize late deliveries through seller-specific ratings rather than abandoning the marketplace altogether—though the low baseline repeat rate (~`3.5%`) also limits statistical power for detecting small retention effects.

## 7. Repository Structure

* `olist_delivery_delay_rdd.ipynb`: End-to-end Google Colab notebook combining Python (ETL, 1-to-1 assertions, feature engineering) and R (`fixest`, `rdrobust`, `ggplot2`).
* `rdd_main_results.tex`: Publication-ready LaTeX table of main Sharp RDD estimates exported via `fixest::etable()`.
* `rdd_robustness_results.tex`: Publication-ready LaTeX table of bandwidth, covariate control, and triangular kernel robustness checks.
* `rdd_review_score_plot.png`: High-resolution RDD binscatter and local linear fit plot (`[-10, 10]` days).

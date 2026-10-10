# Dataset and Project Disclaimer

## 1. Project Overview

Personal Financial Digital Twin with Explainable Spending Forecasting is an educational machine-learning project. It forecasts next-month account spending using a trained CatBoost regression model, provides local feature-level explanations via TreeSHAP, offers counterfactual what-if simulations, and includes financial-planning tools. The application is served through a FastAPI backend with an interactive web frontend.

## 2. Educational Purpose

This application is provided for **educational, research, and demonstration purposes only**.

- It is not a bank, financial institution, or licensed financial services provider.
- It is not a certified financial adviser or planner.
- It does not constitute, and must not be treated as, a substitute for independent professional financial advice.
- It is intended to demonstrate machine-learning techniques, explainability methods, and full-stack application development in a financial domain context.

## 3. Third-Party Dataset Ownership and Attribution

This project uses the **PKDD'99 Czech Financial Dataset**, based on data originally prepared by **Petr Berka** and **Marta Sochorová** for the PKDD 1999 Discovery Challenge.

The project incorporates a publicly hosted modified version of the dataset where applicable. The following sources provide further information:

- **Dataset repository:** [https://github.com/dnoeth/1999_Czech_financial_dataset_Teradata](https://github.com/dnoeth/1999_Czech_financial_dataset_Teradata)
- **Dataset information:** [https://relational.fel.cvut.cz/dataset/Financial](https://relational.fel.cvut.cz/dataset/Financial)

The project author **did not create and does not claim ownership** of the original third-party dataset. All third-party dataset materials are attributed to their respective owners or sources, subject to verification against the original references listed above.

## 4. Dataset Modifications and Limitations

The publicly hosted version of the dataset describes modifications to dates and transaction amounts relative to the original PKDD'99 dataset. The original dataset and the modified version should **not** be assumed to be identical.

- **Dataset provenance, accuracy, and completeness** must be checked against the relevant source documentation and applicable terms.
- **Redistribution rights** for the dataset have not been independently verified by the project author. Users must review the applicable source terms before copying, redistributing, or reusing dataset files.
- This document does not invent, assert, or imply any specific licence for the dataset beyond what is stated by the original dataset authors and hosts.

## 5. Third-Party Rights and Redistribution

- Third-party datasets, software libraries, and other materials incorporated into or used by this project remain subject to the rights and licence terms of their respective owners.
- Users must review the applicable source terms before copying, redistributing, or reusing any third-party materials.
- This document **does not grant permission** to redistribute third-party materials and **does not replace** any required attribution or licence conditions imposed by third-party rights holders.
- The project author does not claim ownership of any third-party material.

## 6. Software and Model Disclaimer

To the extent permitted by applicable law, the application, its predictions, explanations, simulations, generated reports, and all other outputs are provided on an **"as is"** and **"as available"** basis.

- No guarantee is made that predictions, explanations, simulations, exports, or any other results will be accurate, complete, timely, available, or suitable for any particular purpose.
- Machine-learning predictions are statistical estimates derived from historical data patterns. They are **not guarantees** of future financial behaviour or outcomes.
- Model explanations (SHAP values) describe how input features influenced a specific model prediction. They do not represent causal financial relationships.
- Simulation and planning outputs are hypothetical scenarios based on model behaviour, not assured financial projections.

## 7. Financial Decisions

- Outputs from this application **must not be treated as personalised professional financial advice** or as the sole basis for any financial decision.
- Users should independently evaluate all results, consider their own circumstances, and consult a qualified financial, legal, or tax professional where appropriate before acting on any information produced by this application.

## 8. Limitation of Liability

To the extent permitted by applicable law:

- The project author does not guarantee uninterrupted, error-free, or secure operation of the application or any of its components.
- The project author shall not be held liable for any direct, indirect, incidental, consequential, or other damages arising from the use of, reliance on, or inability to use the application, its outputs, or associated materials, except where such liability cannot be excluded or limited under applicable law.
- Nothing in this disclaimer is intended to exclude or limit liability that cannot lawfully be excluded or limited, including liability arising from applicable mandatory legal obligations.

## 9. External Dependencies and Hosting

This project relies on third-party software packages, libraries, frameworks, and hosting services. Their availability, functionality, and terms of use may change independently and without notice.

- The project author does not control and is not responsible for the availability, security, or terms of third-party services or dependencies.
- No credentials, secrets, private information, or deployment tokens are disclosed in this document or intended to be disclosed in the project repository.

## 10. Contact and Corrections

If you identify an attribution error, a rights concern, or wish to request a review of any material in this project, please contact the repository owner through the **repository's public issue tracker**.

## 11. Legal Notice

This document is an **informational project notice**. It is not formal legal advice and does not create, modify, or terminate any legal relationship. It should be reviewed against the applicable laws, regulations, and the actual dataset licence terms before relying on it. Users are responsible for ensuring their own compliance with all applicable legal requirements.

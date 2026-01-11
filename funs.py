from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


__all__ = [
    "DSNetwork",
    "DSText",
    "DSStatsmodels",
    "DSSklearn",
    "DSSklearnText",
    "DSGeo",
    "DSImage",
    "DSTorch",

    "ViTEmbedder",
    "SigLIPEmbedder",
    "E5TextEmbedder",

    "print_rows",
    "dot_product",
    "breaks_width",
    "Path"
]


# utils

def dot_product(a, b, is_array=False):
    return (a * b).list.sum()


def _is_vector(x):
    return isinstance(x, (np.ndarray, pd.Series, pl.Series))


def _round_any(x, accuracy, f = np.round):
    if not _is_vector(x):
        x = np.asarray(x)
    return f(x / accuracy) * accuracy


def print_rows(df: pl.DataFrame, n= -1):
    with pl.Config(tbl_rows=n):
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df)


@dataclass
class breaks_width:
    width: float
    offset: float | None = None

    def __call__(self, limits):
        offset = 0 if self.offset is None else self.offset
        start = _round_any(limits[0], self.width, np.floor) + offset
        end = _round_any(limits[1], self.width, np.ceil) + self.width
        dtype = (
            int
            if isinstance(self.width, int) and isinstance(offset, int)
            else float
        )
        return np.arange(start, end, self.width, dtype=dtype)


# network data

class DSNetwork:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSNetwork is a static utility class and cannot be instantiated")

    @staticmethod
    def process(edges_df, directed=False):
        import igraph as ig

        if isinstance(edges_df, pl.DataFrame):
            edge_list = [
                (row["doc_id"], row["doc_id2"])
                for row in edges_df.iter_rows(named=True)
            ]
        else:
            edge_list = [
                (row["doc_id"], row["doc_id2"])
                for _, row in edges_df.iterrows()
            ]

        G = ig.Graph.TupleList(edge_list, directed=directed)
        layout = G.layout_fruchterman_reingold()
        components = G.connected_components()
        clusters = G.community_walktrap().as_clustering().membership

        vertex_names = [v["name"] for v in G.vs]
        name_to_idx = {name: i for i, name in enumerate(vertex_names)}

        nodes = []
        for i, vertex in enumerate(G.vs):
            component_id = None
            for comp_idx, component in enumerate(components):
                if i in component:
                    component_id = comp_idx + 1
                    break

            node_data = {
                "id": vertex["name"],
                "x": layout[i][0],
                "y": layout[i][1],
                "component": component_id,
                "component_size": len(components[component_id - 1]) if component_id else 0,
                "cluster": str(clusters[i])
            }

            if directed:
                node_data.update({
                    "degree_out": vertex.outdegree(),
                    "degree_in": vertex.indegree(),
                    "degree_total": vertex.degree()
                })
            else:
                node_data["degree"] = vertex.degree()

            nodes.append(node_data)

        eigen_vals = [None] * len(G.vs)
        between_vals = [None] * len(G.vs)
        close_vals = [None] * len(G.vs) if not directed else None

        for comp_idx, component in enumerate(components):
            if len(component) > 1:
                subgraph = G.subgraph(component)
                sub_names = [v["name"] for v in subgraph.vs]

                eigen_scores = subgraph.eigenvector_centrality()
                between_scores = subgraph.betweenness(directed=directed)

                for sub_i, name in enumerate(sub_names):
                    main_i = name_to_idx[name]
                    eigen_vals[main_i] = eigen_scores[sub_i]
                    between_vals[main_i] = between_scores[sub_i]

                if not directed:
                    close_scores = subgraph.closeness()
                    for sub_i, name in enumerate(sub_names):
                        main_i = name_to_idx[name]
                        close_vals[main_i] = close_scores[sub_i]

        for i, node in enumerate(nodes):
            node["eigen"] = eigen_vals[i]
            node["between"] = between_vals[i]
            if not directed:
                node["close"] = close_vals[i]

        node_df = pl.DataFrame(nodes)

        edges_plot = []
        for edge in G.es:
            source_idx = edge.source
            target_idx = edge.target
            edges_plot.append({
                "x": layout[source_idx][0],
                "y": layout[source_idx][1],
                "xend": layout[target_idx][0],
                "yend": layout[target_idx][1]
            })

        edge_df = pl.DataFrame(edges_plot)

        return node_df, edge_df, G


# textual data

class DSText:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSText is a static utility class and cannot be instantiated")

    @staticmethod
    def process(docs_df, nlp):
        tokens_list = []

        for row in docs_df.iter_rows(named=True):
            doc_id = row["doc_id"]
            text = row["text"]
            doc = nlp(text)

            sent_id = 0

            for sent in doc.sents:
                sent_id += 1
                sent_start = sent.start  # doc-global start offset for this sentence

                for token in sent:
                    tid = (token.i - sent_start) + 1  # 1-based within sentence

                    # head index in the SAME coordinate system as tid
                    if token.head == token:  # root
                        head_tid = tid         # or 0 if you want
                    else:
                        head_tid = (token.head.i - sent_start) + 1

                    tokens_list.append({
                        "doc_id": doc_id,
                        "sid": sent_id,
                        "tid": tid,
                        "token": token.text,
                        "token_with_ws": token.text_with_ws,
                        "lemma": token.lemma_,
                        "upos": token.pos_,
                        "tag": token.tag_,
                        "is_alpha": token.is_alpha,
                        "is_stop": token.is_stop,
                        "is_punct": token.is_punct,
                        "dep": token.dep_,
                        "head_idx": head_tid,
                        "ent_type": token.ent_type_,
                        "ent_iob": token.ent_iob_,
                    })

        return pl.DataFrame(tokens_list)

    @staticmethod
    def compute_tfidf(df, min_df=0.0, max_df=1.0):
        """
        min_df / max_df are proportions in [0, 1], e.g.:
          max_df=0.5 keeps terms occurring in <= 50% of documents.
        """

        n_docs = df.select(pl.col("doc_id").n_unique()).item()

        tfidf = (
            df
            .filter(pl.col("is_alpha"))
            .group_by(["doc_id", "lemma"])
            .agg(tf=pl.len())
            .with_columns(
                tf_norm=pl.col("tf") / pl.col("tf").sum().over("doc_id"),
            )
            .with_columns(
                df_docs=pl.col("doc_id").n_unique().over("lemma"),
            )
            .with_columns(
                df_prop=pl.col("df_docs") / pl.lit(n_docs),
            )
            .filter(
                (pl.col("df_prop") >= pl.lit(min_df)) &
                (pl.col("df_prop") <= pl.lit(max_df))
            )
            .with_columns(
                idf=((pl.lit(n_docs) + 1) / (pl.col("df_docs") + 1)).log() + 1,
            )
            .with_columns(
                tfidf=pl.col("tf_norm") * pl.col("idf"),
            )
            .drop("df_prop")
        )

        return tfidf

    @staticmethod
    def kwic(tokens, keyword, max_num=10, window=5, left_width=40, right_width=40):

        keyword = keyword.lower()

        prefix_width = (
            tokens
            .select(pl.col("doc_id").cast(pl.Utf8).str.len_chars().max())
            .item()
            + 6
        )

        matches = tokens.filter(pl.col("lemma") == keyword)
        if len(matches) > max_num:
            matches = matches[:max_num]

        for row in matches.iter_rows(named=True):
            doc = row["doc_id"]
            sid = row["sid"]
            tid = row["tid"]

            context = (
                tokens
                .filter(
                    (pl.col("doc_id") == doc) &
                    (pl.col("sid") == sid) &
                    (pl.col("tid") >= tid - window) &
                    (pl.col("tid") <= tid + window)
                )
                .sort("tid")
            )

            left = (
                context
                .filter(pl.col("tid") < tid)
                .select("token")
                .to_series()
                .to_list()
            )
            right = (
                context
                .filter(pl.col("tid") > tid)
                .select("token")
                .to_series()
                .to_list()
            )

            left_text = " ".join(left)[-left_width:]
            right_text = " ".join(right)[:right_width]

            prefix = f"{doc}:{sid}"

            print(
                f"{prefix:<{prefix_width}} "
                f"{left_text:>{left_width}} "
                f"[{row['token']}] "
                f"{right_text:<{right_width}}"
            )


# image data

class DSImage:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSImage is a static utility class and cannot be instantiated")

    @staticmethod
    def compute_colors(img_hsv):
        import cv2
        h, s, v = cv2.split(img_hsv)
        valid = (s >= 50) & (v >= 50)
        valid.mean()

        color_ranges = {
            'red': ((h <= 10) | (h >= 170)) & valid,
            'orange': (h > 10) & (h <= 25) & valid,
            'yellow': (h > 25) & (h <= 35) & valid,
            'green': (h > 35) & (h <= 85) & valid,
            'cyan': (h > 85) & (h <= 100) & valid,
            'blue': (h > 100) & (h <= 130) & valid,
            'purple': (h > 130) & (h <= 145) & valid,
            'magenta': (h > 145) & (h < 170) & valid,
        }

        total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

        percentages = {color: np.sum(mask) / total_pixels * 100
                       for color, mask in color_ranges.items()}

        percentages['neutral'] = np.sum(~valid) / total_pixels * 100
        return percentages

    @staticmethod
    def plot_image_grid(df, ncol=10, label_name="label", filepath="filepath", limit=100):
        import matplotlib.pyplot as plt
        from PIL import Image
        import math

        df = df.head(limit)
        n = df.height
        if n == 0:
            return

        nrow = math.ceil(n / ncol)

        paths = df.select(filepath).to_series().to_list()
        labels = None
        if label_name is not None and label_name in df.columns:
            labels = df.select(label_name).to_series().to_list()


        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = np.array(axes).ravel()

        for i, ax in enumerate(axes):
            if i < n:
                img = Image.open(paths[i])
                cmap = "gray" if img.mode == "L" else None
                ax.imshow(img, cmap=cmap)

                if labels is not None:
                    ax.set_title(str(labels[i]), fontsize=8)

                w, h = img.size
                max_dim = max(h, w)
                ax.set_xlim(-0.5 + (w - max_dim) / 2, w - 0.5 + (max_dim - w) / 2)
                ax.set_ylim(h - 0.5 + (max_dim - h) / 2, -0.5 + (h - max_dim) / 2)
                ax.set_aspect("equal")
                ax.axis("off")
            else:
                ax.axis("off")

        fig.set_constrained_layout(True)
        plt.show()

    @staticmethod
    def prepare_yolo_dataset(
        df,
        root: str,
        yaml_name: str,
        class_name: str = "label",
        filepath_col: str = "filepath",
        split_col: str = "index",
        bbox_x0_col: str = "bbox_x0",
        bbox_y0_col: str = "bbox_y0",
        bbox_x1_col: str = "bbox_x1",
        bbox_y1_col: str = "bbox_y1",
    ):

        from PIL import Image

        def to_yolo_xywh(x0, y0, x1, y1, w, h):
            xc = ((x0 + x1) / 2) / w
            yc = ((y0 + y1) / 2) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            return xc, yc, bw, bh

        ROOT = Path(root)
        IMG_DIR = ROOT / "images"
        LBL_DIR = ROOT / "labels"

        (IMG_DIR / "train").mkdir(parents=True, exist_ok=True)
        (IMG_DIR / "val").mkdir(parents=True, exist_ok=True)
        (LBL_DIR / "train").mkdir(parents=True, exist_ok=True)
        (LBL_DIR / "val").mkdir(parents=True, exist_ok=True)

        for r in df.iter_rows(named=True):
            split = "train" if r[split_col] == "train" else "val"
            src = Path(r[filepath_col])
            dst_img = IMG_DIR / split / src.name
            dst_lbl = LBL_DIR / split / (src.stem + ".txt")

            if not dst_img.exists():
                import shutil
                shutil.copy2(src, dst_img)

            w, h = Image.open(src).size
            xc, yc, bw, bh = to_yolo_xywh(
                r[bbox_x0_col], r[bbox_y0_col], r[bbox_x1_col], r[bbox_y1_col], w, h
            )

            dst_lbl.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Write YAML config
        data_yaml = ROOT / yaml_name
        data_yaml.write_text(
            f"""path: {ROOT.resolve()}
    train: images/train
    val: images/val
    names:
      0: {class_name}
    """
        )

        return data_yaml

# statistical inference

def _parse_formula_vars(formula):
    from patsy import ModelDesc

    desc = ModelDesc.from_formula(formula)
    lhs_vars = []
    for term in desc.lhs_termlist:
        for factor in term.factors:
            lhs_vars.append(factor.code)
    rhs_vars = []
    for term in desc.rhs_termlist:
        for factor in term.factors:
            rhs_vars.append(factor.code)
    return lhs_vars, rhs_vars


def _simple_table_to_polars(tbl):
    from io import StringIO

    csv_str = tbl.as_csv()
    cleaned = "\n".join(
        ",".join(field.strip() for field in line.split(","))
        for line in csv_str.splitlines()
    )
    return pl.read_csv(StringIO(cleaned))


class DSStatsmodels:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSStatsmodels is a static utility class and cannot be instantiated")

    @staticmethod
    def ttest1(df, formula):
        from patsy import dmatrices
        import statsmodels.api as sm

        pdf = df.to_pandas()
        y, _ = dmatrices(formula, data=pdf, return_type="dataframe")
        y = y.iloc[:, 0]
        descr = sm.stats.DescrStatsW(y)
        t_stat, p_value, df_val = descr.ttest_mean(value=0.0)
        ci_low, ci_high = descr.tconfint_mean(alpha=0.05)
        return pl.DataFrame({
            "mean": [y.mean()],
            "t": [float(t_stat)],
            "P>|t|": [float(p_value)],
            "[0.025": [float(ci_low)],
            "0.975]": [float(ci_high)],
        })

    @staticmethod
    def ttest2(df, formula):
        import statsmodels.formula.api as smf
        import re

        pdf = df.to_pandas()
        model = smf.ols(formula=formula, data=pdf)
        result = model.fit()
        param_index = list(result.params.index)
        intercept_names = {"intercept", "const"}
        non_intercepts = [name for name in param_index if name.lower() not in intercept_names]
        if len(non_intercepts) != 1:
            raise ValueError(
                f"Expected exactly one non-intercept parameter (binary factor), "
                f"found {len(non_intercepts)}: {non_intercepts}"
            )
        term = non_intercepts[0]
        match = re.match(r"(?:C\()?(\w+)\)?(?:\[T\.(.+)\])?", term)
        if match:
            group_var = match.group(1)
            comparison_level = match.group(2)
        else:
            group_var = term
            comparison_level = None
        levels = sorted(pdf[group_var].unique())
        if len(levels) != 2:
            raise ValueError(f"Expected exactly 2 levels, found {len(levels)}: {levels}")
        level1 = levels[0]
        level2 = levels[1]
        lhs_vars, _ = _parse_formula_vars(formula)
        response_var = lhs_vars[0]
        mean1 = float(pdf[pdf[group_var] == level1][response_var].mean())
        mean2 = float(pdf[pdf[group_var] == level2][response_var].mean())
        estimate = float(result.params[term])
        statistic = float(result.tvalues[term])
        p_value = float(result.pvalues[term])
        ci = result.conf_int(alpha=0.05)
        ci_lower = float(ci.loc[term, 0])
        ci_upper = float(ci.loc[term, 1])
        return pl.DataFrame({
            "level1": [level1],
            "level2": [level2],
            "mean1": [mean1],
            "mean2": [mean2],
            "mean_diff": [estimate],
            "t": [statistic],
            "P>|t|": [p_value],
            "[0.025": [ci_lower],
            "0.975]": [ci_upper],
        })

    @staticmethod
    def anova(df, formula):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        pdf = df.to_pandas()
        model = smf.ols(formula=formula, data=pdf)
        result = model.fit()
        anova_tbl = anova_lm(result, typ=2)
        anova_tbl = anova_tbl.reset_index()
        return pl.DataFrame(anova_tbl)

    @staticmethod
    def chi2(df, formula):
        from scipy.stats import chi2_contingency

        pdf = df.to_pandas()
        lhs_vars, rhs_vars = _parse_formula_vars(formula)
        row_var = lhs_vars[0]
        col_var = rhs_vars[0]
        contingency = pdf.groupby([row_var, col_var]).size().unstack(fill_value=0)
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        return pl.DataFrame({
            "χ²": [float(chi2)],
            "df": [int(dof)],
            "P>|χ²|": [float(p_value)]
        })

    @staticmethod
    def gtest(df, formula):
        from scipy.stats import chi2_contingency

        pdf = df.to_pandas()
        lhs_vars, rhs_vars = _parse_formula_vars(formula)
        row_var = lhs_vars[0]
        col_var = rhs_vars[0]
        contingency = pdf.groupby([row_var, col_var]).size().unstack(fill_value=0)
        g_stat, p_value, dof, expected = chi2_contingency(contingency, lambda_="log-likelihood")
        return pl.DataFrame({
            "G": [float(g_stat)],
            "df": [int(dof)],
            "P>|G|": [float(p_value)],
        })

    @staticmethod
    def ols(df, formula, raw=False, **kwargs):
        import statsmodels.formula.api as smf

        pdf = df.to_pandas()
        model = smf.ols(formula=formula, data=pdf, **kwargs).fit(disp=False)
        if raw:
            return model
        return _simple_table_to_polars(model.summary().tables[1])

    @staticmethod
    def wls(df, formula, raw=False, **kwargs):
        import statsmodels.formula.api as smf

        pdf = df.to_pandas()
        model = smf.wls(formula=formula, data=pdf, **kwargs).fit(disp=False)
        if raw:
            return model
        return _simple_table_to_polars(model.summary().tables[1])

    @staticmethod
    def glm(df, formula, raw=False, **kwargs):
        import statsmodels.formula.api as smf

        pdf = df.to_pandas()
        model = smf.glm(formula=formula, data=pdf, **kwargs).fit(disp=False)
        if raw:
            return model
        return _simple_table_to_polars(model.summary().tables[1])

    @staticmethod
    def logit(df, formula, raw=False, **kwargs):
        import statsmodels.formula.api as smf

        pdf = df.to_pandas()
        model = smf.logit(formula=formula, data=pdf, **kwargs).fit(disp=False)
        if raw:
            return model
        return _simple_table_to_polars(model.summary().tables[1])


# sklearn

class SkWrapperClass:
    def __init__(self, model_name, model, df, X, y, index, train_idx, test_idx, X_train, X_test, y_train, y_test, column_names):
        self.model_name = model_name
        self.model = model
        self.df = df
        self.X = X
        self.y = y
        self.index = index
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.column_names = column_names

    def predict(self, full=False):
        predictions = self.model.predict(self.X)
        index_labels = np.empty(len(self.X), dtype=object)
        index_labels[self.train_idx] = "train"
        index_labels[self.test_idx] = "test"
        result = pl.DataFrame({
            "index_": index_labels.tolist(),
            "target_": self.y.tolist(),
            "prediction_": predictions.tolist()
        })
        if full:
            return pl.concat([self.df, result], how="horizontal")
        return result

    def predict_proba(self, full=False):
        if self.model_name not in ["logistic_regression", "logistic_regression_cv", "gradient_boosting_classifier", "random_forest_classifier"]:
            raise ValueError(f"Model '{self.model_name}' is not a classifier. Use logistic_regression, logistic_regression_cv, gradient_boosting_classifier, or random_forest_classifier.")
        predictions = self.model.predict(self.X)
        probabilities = self.model.predict_proba(self.X)
        max_probabilities = probabilities.max(axis=1)
        index_labels = np.empty(len(self.X), dtype=object)
        index_labels[self.train_idx] = "train"
        index_labels[self.test_idx] = "test"
        classes = self.model.named_steps["model"].classes_
        data = {
            "index_": index_labels.tolist(),
            "target_": self.y.tolist(),
            "prediction_": predictions.tolist(),
            "prob_pred_": max_probabilities.tolist()
        }
        for i, cls in enumerate(classes):
            data[str(cls)] = probabilities[:, i].tolist()
        result = pl.DataFrame(data)
        if full:
            return pl.concat([self.df, result], how="horizontal")
        return result

    def coef(self, raw=False):
        if self.model_name not in ["linear_regression", "elastic_net", "elastic_net_cv", "logistic_regression", "logistic_regression_cv"]:
            raise ValueError(f"Model '{self.model_name}' does not have coefficients. Use linear_regression, elastic_net, elastic_net_cv, logistic_regression, or logistic_regression_cv.")
        fitted_model = self.model.named_steps["model"]
        scaler = self.model.named_steps["scaler"]

        if raw:
            scaled_coef = fitted_model.coef_
            coef = scaled_coef / scaler.scale_
            intercept = fitted_model.intercept_ - np.dot(coef, scaler.mean_)
        else:
            coef = fitted_model.coef_
            intercept = fitted_model.intercept_

        if coef.ndim == 2:
            classes = fitted_model.classes_
            data = {"name": list(self.column_names)}
            for i, cls in enumerate(classes):
                data[str(cls)] = list(coef[i])
            df = pl.DataFrame(data)

            intercept_data = {"name": ["Intercept"]}
            for i, cls in enumerate(classes):
                intercept_data[str(cls)] = [intercept[i]]
            intercept_row = pl.DataFrame(intercept_data)
            return pl.concat([intercept_row, df])
        else:
            df = pl.DataFrame({
                "name": list(self.column_names),
                "param": list(coef)
            }).sort("param", descending=True)

            intercept_row = pl.DataFrame({
                "name": ["Intercept"],
                "param": [intercept]
            })
            return pl.concat([intercept_row, df])

    def importance(self):
        if self.model_name not in ["gradient_boosting_classifier", "gradient_boosting_regressor", "random_forest_classifier", "random_forest_regressor"]:
            raise ValueError(f"Model '{self.model_name}' does not have feature importances. Use gradient_boosting_classifier, gradient_boosting_regressor, random_forest_classifier, or random_forest_regressor.")
        fitted_model = self.model.named_steps["model"]
        return pl.DataFrame({
            "name": list(self.column_names),
            "importance": fitted_model.feature_importances_.tolist()
        })

    def alpha(self):
        if self.model_name not in ["elastic_net_cv", "logistic_regression_cv"]:
            raise ValueError(f"Model '{self.model_name}' does not have a CV-selected alpha. Use elastic_net_cv or logistic_regression_cv.")
        fitted_model = self.model.named_steps["model"]
        if self.model_name == "elastic_net_cv":
            return fitted_model.alpha_
        else:
            return fitted_model.C_

    def score(self):
        regression_models = ["linear_regression", "elastic_net", "elastic_net_cv", "gradient_boosting_regressor", "random_forest_regressor"]
        if self.model_name in regression_models:
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            train_rmse = np.sqrt(np.mean((self.y[self.train_idx] - y_train_pred) ** 2))
            test_rmse = np.sqrt(np.mean((self.y[self.test_idx] - y_test_pred) ** 2))
            return {"train": train_rmse, "test": test_rmse}
        else:
            return {
                "train": self.model.score(self.X_train, self.y[self.train_idx]),
                "test": self.model.score(self.X_test, self.y[self.test_idx])
            }

    def confusion_matrix(self, kind="test"):
        from sklearn.metrics import ConfusionMatrixDisplay

        if self.model_name not in ["logistic_regression", "logistic_regression_cv", "gradient_boosting_classifier", "random_forest_classifier"]:
            raise ValueError(f"Model '{self.model_name}' is not a classifier. Use logistic_regression, logistic_regression_cv, gradient_boosting_classifier, or random_forest_classifier.")
        if kind == "test":
            y_true = self.y[self.test_idx]
            y_pred = self.model.predict(self.X_test)
        elif kind == "train":
            y_true = self.y[self.train_idx]
            y_pred = self.model.predict(self.X_train)
        elif kind == "all":
            y_true = self.y
            y_pred = self.model.predict(self.X)
        else:
            raise ValueError(f"kind must be 'test', 'train', or 'all', got '{kind}'")
        return ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")


class SkWrapperDimred:
    def __init__(self, model_name, model, df, X, column_names, embedding):
        self.model_name = model_name
        self.model = model
        self.df = df
        self.X = X
        self.column_names = column_names
        self.embedding = embedding

    def predict(self, prefix="dr", array=False, full=False):
        data = {}
        if array:
            data[prefix] = self.embedding.tolist()
            result = pl.DataFrame(data).cast({prefix: pl.Array(pl.Float64, self.embedding.shape[1])})
        else:
            n_components = self.embedding.shape[1]
            pad_width = len(str(n_components - 1))
            for i in range(n_components):
                col_name = f"{prefix}{str(i).zfill(pad_width)}"
                data[col_name] = self.embedding[:, i].tolist()
            result = pl.DataFrame(data)

        if full:
            return pl.concat([self.df, result], how="horizontal")
        return result


class SkWrapperCluster:
    def __init__(self, model_name, model, df, X, column_names, labels):
        self.model_name = model_name
        self.model = model
        self.df = df
        self.X = X
        self.column_names = column_names
        self.labels = labels

    def predict(self, full=False):
        labels = self.labels
        n_samples = len(labels)
        distances = np.zeros(n_samples)

        if self.model_name == "kmeans":
            all_distances = self.model.transform(self.X)
            for i in range(n_samples):
                distances[i] = all_distances[i, labels[i]]
        elif self.model_name == "dbscan":
            unique_labels = set(labels) - {-1}
            centroids = {}
            for label in unique_labels:
                mask = labels == label
                centroids[label] = self.X[mask].mean(axis=0)

            for i in range(n_samples):
                if labels[i] == -1:
                    distances[i] = np.nan
                else:
                    distances[i] = np.linalg.norm(self.X[i] - centroids[labels[i]])

        result = pl.DataFrame({
            "label_": labels.tolist(),
            "dist_": distances.tolist()
        })

        if full:
            return pl.concat([self.df, result], how="horizontal")
        return result


def _extract_features(df, target_name, features, drop):
    if features is not None:
        X_df = df.select(features)
    elif drop is not None:
        drop_names = df.select(drop).columns
        exclude = set(drop_names)
        if target_name is not None:
            exclude.add(target_name)
        all_columns = [col for col in df.columns if col not in exclude]
        X_df = df.select(all_columns)
    else:
        if target_name is not None:
            all_columns = [col for col in df.columns if col != target_name]
        else:
            all_columns = df.columns
        X_df = df.select(all_columns)

    if len(X_df.columns) == 1:
        col_dtype = X_df.dtypes[0]
        if isinstance(col_dtype, (pl.List, pl.Array)):
            X = np.vstack(X_df.to_series().to_list())
            n_cols = X.shape[1]
            pad_width = len(str(n_cols - 1))
            column_names = [f"col{str(i).zfill(pad_width)}" for i in range(n_cols)]
        else:
            column_names = X_df.columns
            X = X_df.to_numpy()
    else:
        column_names = X_df.columns
        X = X_df.to_numpy()

    return X, column_names


def _fit_supervised(model_name, X, y, df, column_names, test_size, random_state, stratify_array, **kwargs):
    from sklearn.model_selection import train_test_split

    index = np.arange(X.shape[0])

    train_idx, test_idx = train_test_split(
        index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_array
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        ElasticNet,
        ElasticNetCV,
        LinearRegression,
        LogisticRegression,
        LogisticRegressionCV,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SUPPORTED_MODELS = {
        "linear_regression": LinearRegression,
        "elastic_net": ElasticNet,
        "elastic_net_cv": ElasticNetCV,
        "logistic_regression": LogisticRegression,
        "logistic_regression_cv": LogisticRegressionCV,
        "gradient_boosting_classifier": GradientBoostingClassifier,
        "gradient_boosting_regressor": GradientBoostingRegressor,
        "random_forest_classifier": RandomForestClassifier,
        "random_forest_regressor": RandomForestRegressor,
    }

    model_class = SUPPORTED_MODELS[model_name]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_class(**kwargs))
    ])

    pipeline.fit(X_train, y_train)

    return SkWrapperClass(
        model_name=model_name,
        model=pipeline,
        df=df,
        X=X,
        y=y,
        index=index,
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        column_names=column_names
    )


def _fit_dimred(model_name, X, df, column_names, scale=True, **kwargs):
    from sklearn.preprocessing import StandardScaler

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    import umap
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE

    DIMRED_MODELS = {
        "pca": PCA,
        "tsne": TSNE,
        "umap": umap.UMAP,
        "tsvd": TruncatedSVD
    }

    model_class = DIMRED_MODELS[model_name]
    model = model_class(**kwargs)
    embedding = model.fit_transform(X_scaled)

    return SkWrapperDimred(
        model_name=model_name,
        model=model,
        df=df,
        X=X_scaled,
        column_names=column_names,
        embedding=embedding
    )


def _fit_cluster(model_name, X, df, column_names, **kwargs):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.cluster import DBSCAN, KMeans

    CLUSTER_MODELS = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
    }

    model_class = CLUSTER_MODELS[model_name]
    model = model_class(**kwargs)
    model.fit(X_scaled)
    labels = model.labels_

    return SkWrapperCluster(
        model_name=model_name,
        model=model,
        df=df,
        X=X_scaled,
        column_names=column_names,
        labels=labels
    )


def _build_dtm(
    df,
    doc_id,
    term_id,
    count=None,
    max_vocab_size=None,
    min_df=0.0,
    max_df=1.0,
):
    """
    Build a document-term matrix returning TF-IDF weights.

    min_df / max_df are proportions in [0, 1] (document frequency fraction).
    """

    doc_col = df.select(doc_id).columns[0]
    term_col = df.select(term_id).columns[0]

    if count is not None:
        count_col = df.select(count).columns[0]
        dtm_df = df.select([doc_col, term_col, count_col])
    else:
        count_col = "_count_"
        dtm_df = df.select([doc_col, term_col]).with_columns(pl.lit(1).alias(count_col))

    # Stable doc index
    unique_docs = dtm_df.select(doc_col).unique().sort(doc_col)
    doc_ids = unique_docs.to_series().to_list()
    n_docs = len(doc_ids)

    if n_docs == 0:
        raise ValueError("No documents found.")

    # Document frequency per term (how many docs contain the term)
    term_doc_counts = (
        dtm_df
        .select([doc_col, term_col])
        .unique()
        .group_by(term_col)
        .agg(pl.len().alias("df_docs"))
        .with_columns((pl.col("df_docs") / pl.lit(n_docs)).alias("df_prop"))
    )

    # Apply df proportion filters
    valid_terms = term_doc_counts.filter(
        (pl.col("df_prop") >= pl.lit(min_df)) & (pl.col("df_prop") <= pl.lit(max_df))
    )

    # Optional vocab cap by df (common heuristic); you can switch to other criteria if preferred
    if max_vocab_size is not None and valid_terms.height > max_vocab_size:
        valid_terms = valid_terms.sort("df_docs", descending=True).head(max_vocab_size)

    valid_term_set = set(valid_terms.select(term_col).to_series().to_list())

    dtm_df = dtm_df.filter(pl.col(term_col).is_in(valid_term_set))

    # Stable term index
    unique_terms = dtm_df.select(term_col).unique().sort(term_col)
    term_names = unique_terms.to_series().to_list()
    n_terms = len(term_names)

    if n_terms == 0:
        raise ValueError("No terms remaining after applying min_df and max_df filters.")

    doc_to_idx = {doc: idx for idx, doc in enumerate(doc_ids)}
    term_to_idx = {term: idx for idx, term in enumerate(term_names)}

    # Aggregate counts per (doc, term)
    aggregated = (
        dtm_df
        .group_by([doc_col, term_col])
        .agg(pl.col(count_col).sum().alias("tf"))
    )

    # Compute normalized tf per doc
    aggregated = aggregated.with_columns(
        (pl.col("tf") / pl.col("tf").sum().over(doc_col)).alias("tf_norm")
    )

    # Join df_docs (document frequency) onto aggregated
    aggregated = aggregated.join(
        valid_terms.select([term_col, "df_docs"]),
        on=term_col,
        how="inner",
    )

    # Compute smoothed idf and tf-idf
    aggregated = aggregated.with_columns(
        ( ((pl.lit(n_docs) + 1) / (pl.col("df_docs") + 1)).log() + 1 ).alias("idf")
    ).with_columns(
        (pl.col("tf_norm") * pl.col("idf")).alias("tfidf")
    )

    # Build sparse matrix from tfidf weights
    rows, cols, data = [], [], []
    for row in aggregated.select([doc_col, term_col, "tfidf"]).iter_rows(named=True):
        doc = row[doc_col]
        term = row[term_col]
        val = float(row["tfidf"])

        # Guard (should always be true)
        if doc in doc_to_idx and term in term_to_idx:
            rows.append(doc_to_idx[doc])
            cols.append(term_to_idx[term])
            data.append(val)

    from scipy.sparse import csr_matrix

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))
    X = sparse_matrix.toarray()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    return X, doc_ids, term_names


def _prepare_dtm_target(df, doc_id, target, doc_ids, is_supervised):
    doc_col = df.select(doc_id).columns[0]
    doc_df = pl.DataFrame({doc_col: doc_ids})

    y = None

    if target is not None:
        target_col = df.select(target).columns[0]

        doc_targets = (
            df
            .select([doc_col, target_col])
            .unique()
            .sort(doc_col)
        )

        if len(doc_targets) != len(doc_ids):
            raise ValueError(f"Target variable must have exactly one value per document. "
                           f"Found {len(doc_targets)} unique (doc_id, target) pairs for {len(doc_ids)} documents.")

        target_map = dict(zip(
            doc_targets.select(doc_col).to_series().to_list(),
            doc_targets.select(target_col).to_series().to_list()
        ))
        y = np.array([target_map[doc] for doc in doc_ids])

        doc_df = doc_df.with_columns(pl.Series(name=target_col, values=y))

    elif is_supervised:
        raise ValueError("Supervised models require a target variable.")

    return doc_df, y


def _prepare_dtm_stratify(df, doc_id, stratify, doc_ids):
    if stratify is None:
        return None

    doc_col = df.select(doc_id).columns[0]
    stratify_col = df.select(stratify).columns[0]

    doc_stratify = (
        df
        .select([doc_col, stratify_col])
        .unique()
        .sort(doc_col)
    )

    if len(doc_stratify) != len(doc_ids):
        raise ValueError(f"Stratify variable must have exactly one value per document.")

    stratify_map = dict(zip(
        doc_stratify.select(doc_col).to_series().to_list(),
        doc_stratify.select(stratify_col).to_series().to_list()
    ))
    return np.array([stratify_map[doc] for doc in doc_ids])


class DSSklearn:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSSklearn is a static utility class and cannot be instantiated")

    @staticmethod
    def linear_regression(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("linear_regression", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def elastic_net(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("elastic_net", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def elastic_net_cv(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("elastic_net_cv", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def logistic_regression(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("logistic_regression", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def logistic_regression_cv(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("logistic_regression_cv", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def gradient_boosting_classifier(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("gradient_boosting_classifier", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def gradient_boosting_regressor(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("gradient_boosting_regressor", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def random_forest_classifier(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("random_forest_classifier", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def random_forest_regressor(df, target, features=None, drop=None, test_size=0.3, random_state=0, stratify=None, **kwargs):
        target_name = df.select(target).columns[0]
        y = df.select(target).to_numpy().ravel()
        X, column_names = _extract_features(df, target_name, features, drop)
        stratify_array = df.select(stratify).to_numpy().ravel() if stratify is not None else None
        return _fit_supervised("random_forest_regressor", X, y, df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def pca(df, features=None, drop=None, **kwargs):
        X, column_names = _extract_features(df, None, features, drop)
        return _fit_dimred("pca", X, df, column_names, **kwargs)

    @staticmethod
    def tsne(df, features=None, drop=None, **kwargs):
        X, column_names = _extract_features(df, None, features, drop)
        return _fit_dimred("tsne", X, df, column_names, **kwargs)

    @staticmethod
    def umap(df, features=None, drop=None, **kwargs):
        X, column_names = _extract_features(df, None, features, drop)
        return _fit_dimred("umap", X, df, column_names, **kwargs)

    @staticmethod
    def kmeans(df, features=None, drop=None, **kwargs):
        X, column_names = _extract_features(df, None, features, drop)
        return _fit_cluster("kmeans", X, df, column_names, **kwargs)

    @staticmethod
    def dbscan(df, features=None, drop=None, **kwargs):
        X, column_names = _extract_features(df, None, features, drop)
        return _fit_cluster("dbscan", X, df, column_names, **kwargs)


class DSSklearnText:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSSklearnText is a static utility class and cannot be instantiated")

    @staticmethod
    def linear_regression(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("linear_regression", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def elastic_net(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("elastic_net", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def elastic_net_cv(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("elastic_net_cv", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def logistic_regression(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("logistic_regression", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def logistic_regression_cv(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("logistic_regression_cv", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def gradient_boosting_classifier(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("gradient_boosting_classifier", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def gradient_boosting_regressor(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("gradient_boosting_regressor", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def random_forest_classifier(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("random_forest_classifier", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def random_forest_regressor(df, doc_id, term_id, target, count=None, max_vocab_size=None, min_df=0, max_df=1, test_size=0.3, random_state=0, stratify=None, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_df, y = _prepare_dtm_target(df, doc_id, target, doc_ids, True)
        stratify_array = _prepare_dtm_stratify(df, doc_id, stratify, doc_ids)
        column_names = [str(t) for t in term_names]
        return _fit_supervised("random_forest_regressor", X, y, doc_df, column_names, test_size, random_state, stratify_array, **kwargs)

    @staticmethod
    def pca(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_dimred("pca", X, doc_df, column_names, scale=False, **kwargs)

    @staticmethod
    def tsvd(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_dimred("tsvd", X, doc_df, column_names, scale=False, **kwargs)

    @staticmethod
    def tsne(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_dimred("tsne", X, doc_df, column_names, **kwargs)

    @staticmethod
    def umap(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_dimred("umap", X, doc_df, column_names, **kwargs)

    @staticmethod
    def kmeans(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_cluster("kmeans", X, doc_df, column_names, **kwargs)

    @staticmethod
    def dbscan(df, doc_id, term_id, count=None, max_vocab_size=None, min_df=0, max_df=1, **kwargs):
        X, doc_ids, term_names = _build_dtm(df, doc_id, term_id, count, max_vocab_size, min_df, max_df)
        doc_col = df.select(doc_id).columns[0]
        doc_df = pl.DataFrame({doc_col: doc_ids})
        column_names = [str(t) for t in term_names]
        return _fit_cluster("dbscan", X, doc_df, column_names, **kwargs)


# geospatial data

DEFAULT_CRS = "EPSG:4326"


def _gpd_to_pl(gdf, geometry_col="geometry"):
    import geopandas as gpd

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")
    if geometry_col not in gdf.columns:
        raise ValueError(f"Missing geometry column '{geometry_col}'")
    if gdf.geometry.name != geometry_col:
        gdf = gdf.set_geometry(geometry_col)

    if gdf.crs is not None and gdf.crs != DEFAULT_CRS:
        gdf = gdf.to_crs(DEFAULT_CRS)

    pdf = gdf.copy()
    pdf[geometry_col] = gdf.geometry.to_wkb()
    pl_df = pl.from_pandas(pdf).with_columns(pl.col(geometry_col).cast(pl.Binary))

    other_cols = [c for c in pl_df.columns if c != geometry_col]
    return pl_df.select(other_cols + [geometry_col])


def _pl_to_gpd(pl_df, geometry_col="geometry"):
    import geopandas as gpd

    if not isinstance(pl_df, pl.DataFrame):
        raise TypeError("pl_df must be a Polars DataFrame")
    if geometry_col not in pl_df.columns:
        raise ValueError(f"Missing geometry column '{geometry_col}'")

    pdf = pl_df.to_pandas()
    geom = gpd.GeoSeries.from_wkb(pdf[geometry_col], crs=DEFAULT_CRS)
    gdf = gpd.GeoDataFrame(pdf, geometry=geom, crs=DEFAULT_CRS)
    if gdf.geometry.name != geometry_col:
        gdf = gdf.rename_geometry(geometry_col)
    return gdf


class DSGeo:
    def __new__(cls, *args, **kwargs):
        raise TypeError("GeoPolars is a static utility class and cannot be instantiated")

    @staticmethod
    def read_file(path, geometry_col="geometry", **kwargs):
        import geopandas as gpd

        gdf = gpd.read_file(path, **kwargs)
        if geometry_col != "geometry":
            gdf = gdf.rename_geometry(geometry_col)
        gdf = gdf.to_crs(DEFAULT_CRS)
        return _gpd_to_pl(gdf, geometry_col=geometry_col)

    @staticmethod
    def from_latlon(pl_df, lat="lat", lon="lon", geometry_col="geometry"):
        import geopandas as gpd

        if geometry_col in pl_df.columns:
            raise ValueError(f"Column '{geometry_col}' already exists")

        lat_expr = lat if isinstance(lat, pl.Expr) else pl.col(lat)
        lon_expr = lon if isinstance(lon, pl.Expr) else pl.col(lon)

        tmp = pl_df.with_columns(
            lon_expr.cast(pl.Float64).alias("__gp_lon__"),
            lat_expr.cast(pl.Float64).alias("__gp_lat__"),
        )

        pdf = tmp.to_pandas()
        gdf = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["__gp_lon__"], pdf["__gp_lat__"]),
            crs=DEFAULT_CRS,
        )
        gdf = gdf.drop(columns=["__gp_lon__", "__gp_lat__"])

        out = _gpd_to_pl(gdf, geometry_col="geometry")
        if geometry_col != "geometry":
            out = out.rename({"geometry": geometry_col})
        return out

    @staticmethod
    def add_centroid(pl_df, geometry_col="geometry", lon_col="lon", lat_col="lat"):
        gdf = _pl_to_gpd(pl_df, geometry_col=geometry_col)
        cent = gdf.geometry.centroid

        return pl_df.with_columns(
            pl.Series(name=lon_col, values=cent.x.to_numpy()).cast(pl.Float64),
            pl.Series(name=lat_col, values=cent.y.to_numpy()).cast(pl.Float64),
        )

    @staticmethod
    def add_area(pl_df, crs=None, geometry_col="geometry", out_col="area"):
        gdf = _pl_to_gpd(pl_df, geometry_col=geometry_col)

        if crs is not None:
            gdf = gdf.to_crs(crs)

        areas = gdf.geometry.area.to_numpy() / 1_000_000
        return pl_df.with_columns(pl.Series(name=out_col, values=areas).cast(pl.Float64))

    @staticmethod
    def plot(
        pl_df,
        geometry_col="geometry",
        crs=None,
        label=None,
        color_by=None,
        cmap="Spectral_r",
        legend=False,
        figsize=(8, 10),
        point_markersize=10,
        polygon_edgecolor="black",
        polygon_linewidth=0.4,
        polygon_facecolor="lightgray",
        show=True,
        ax=None,
    ):
        import matplotlib.pyplot as plt

        gdf = _pl_to_gpd(pl_df, geometry_col=geometry_col)

        if crs is not None:
            gdf = gdf.to_crs(crs)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        geom_type = gdf.geometry.geom_type
        is_polygon = geom_type.isin(["Polygon", "MultiPolygon"]).any()
        is_point = geom_type.isin(["Point", "MultiPoint"]).any()

        plot_kwargs = {}

        if color_by is not None:
            if isinstance(color_by, pl.Expr):
                vals = pl_df.select(color_by.alias("__color__")).get_column("__color__").to_list()
                gdf["__color__"] = vals
                plot_kwargs["column"] = "__color__"
            else:
                if color_by not in gdf.columns:
                    raise ValueError(f"Color column '{color_by}' not found")
                plot_kwargs["column"] = color_by

            plot_kwargs["legend"] = legend
            if cmap is not None:
                plot_kwargs["cmap"] = cmap

        if is_polygon:
            if "column" not in plot_kwargs:
                plot_kwargs["color"] = polygon_facecolor
            plot_kwargs.setdefault("edgecolor", polygon_edgecolor)
            plot_kwargs.setdefault("linewidth", polygon_linewidth)
            gdf.plot(ax=ax, **plot_kwargs)

        if is_point and not is_polygon:
            if "column" not in plot_kwargs:
                gdf.plot(ax=ax, markersize=point_markersize)
            else:
                gdf.plot(ax=ax, markersize=point_markersize, **plot_kwargs)

        if label is not None:
            if isinstance(label, pl.Expr):
                lbl = pl_df.select(label.alias("__label__")).get_column("__label__").to_list()
                gdf["__label__"] = lbl
                label_col = "__label__"
            else:
                label_col = label
                if label_col not in gdf.columns:
                    raise ValueError(f"Label column '{label_col}' not found")

            label_points = gdf.geometry if not is_polygon else gdf.geometry.centroid

            for x, y, t in zip(label_points.x, label_points.y, gdf[label_col]):
                if t is None:
                    continue
                ax.text(x, y, str(t), ha="center", va="center", fontsize=7)

        ax.set_axis_off()

        if show:
            plt.show()

        return ax

    @staticmethod
    def plot_layers(
        layers,
        geometry_col="geometry",
        crs=None,
        figsize=(8, 10),
        show=True,
        ax=None,
        defaults=None,
    ):
        import matplotlib.pyplot as plt

        if not isinstance(layers, (list, tuple)) or len(layers) == 0:
            raise ValueError("layers must be a non-empty list/tuple")

        if defaults is None:
            defaults = {}

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        for item in layers:
            if isinstance(item, tuple) and len(item) == 2:
                layer_df, opts = item
                if not isinstance(opts, dict):
                    raise TypeError("options_dict must be a dict")
            else:
                layer_df, opts = item, {}

            merged = dict(defaults)
            merged.update(opts)

            GeoPolars.plot(
                layer_df,
                geometry_col=merged.get("geometry_col", geometry_col),
                crs=merged.get("crs", crs),
                label=merged.get("label", None),
                color_by=merged.get("color_by", None),
                cmap=merged.get("cmap", None),
                legend=merged.get("legend", False),
                figsize=merged.get("figsize", figsize),
                point_markersize=merged.get("point_markersize", 10),
                polygon_edgecolor=merged.get("polygon_edgecolor", "black"),
                polygon_linewidth=merged.get("polygon_linewidth", 0.4),
                polygon_facecolor=merged.get("polygon_facecolor", "lightgray"),
                show=False,
                ax=ax,
            )

        ax.set_axis_off()
        if show:
            plt.show()
        return ax

    @staticmethod
    def sjoin(
        left,
        right,
        geometry_col="geometry",
        crs=None,
        predicate="intersects",
        how="inner",
        lsuffix="_left",
        rsuffix="_right",
        **kwargs,
    ):
        import geopandas as gpd

        gleft = _pl_to_gpd(left, geometry_col=geometry_col)
        gright = _pl_to_gpd(right, geometry_col=geometry_col)

        if crs is not None:
            gleft = gleft.to_crs(crs)
            gright = gright.to_crs(crs)

        joined = gpd.sjoin(
            gleft,
            gright,
            how=how,
            predicate=predicate,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            **kwargs,
        )

        if crs is not None:
            joined = joined.to_crs(DEFAULT_CRS)

        result = _gpd_to_pl(joined, geometry_col=geometry_col)

        rename_map = {c: c.removesuffix("_" + lsuffix) for c in result.columns if c.endswith(lsuffix)}
        if rename_map:
            result = result.rename(rename_map)

        return result

    @staticmethod
    def sjoin_nearest(
        left,
        right,
        geometry_col="geometry",
        crs=None,
        how="inner",
        max_distance=None,
        distance_col=None,
        lsuffix="left",
        rsuffix="right",
        **kwargs,
    ):
        import geopandas as gpd

        gleft = _pl_to_gpd(left, geometry_col=geometry_col)
        gright = _pl_to_gpd(right, geometry_col=geometry_col)

        if crs is not None:
            gleft = gleft.to_crs(crs)
            gright = gright.to_crs(crs)

        joined = gpd.sjoin_nearest(
            gleft,
            gright,
            how=how,
            max_distance=max_distance,
            distance_col=distance_col,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            **kwargs,
        )

        if crs is not None:
            joined = joined.to_crs(DEFAULT_CRS)

        result = _gpd_to_pl(joined, geometry_col=geometry_col)

        rename_map = {c: c.removesuffix("_" + lsuffix) for c in result.columns if c.endswith(lsuffix)}
        if rename_map:
            result = result.rename(rename_map)

        return result

    @staticmethod
    def explore(
        pl_df,
        geometry_col="geometry",
        column=None,
        tooltip=None,
        tiles="CartoDB positron",
        cmap="Spectral_r",
        style_kwds=None,
        color_identity=False,
        default_color="#808000",
        **kwargs,
    ):
        gdf = _pl_to_gpd(pl_df, geometry_col=geometry_col)

        if style_kwds is None:
            style_kwds = {}

        col_name = None
        if column is not None:
            if isinstance(column, pl.Expr):
                gdf["__gp_color__"] = pl_df.select(column.alias("__gp_color__")).get_column("__gp_color__").to_list()
                col_name = "__gp_color__"
            else:
                col_name = column
                if col_name not in gdf.columns:
                    raise ValueError(f"Column '{col_name}' not found for coloring")

        tooltip_cols = None
        if tooltip is not None:
            if isinstance(tooltip, (list, tuple)):
                tooltip_cols = []
                for t in tooltip:
                    if isinstance(t, pl.Expr):
                        name = t.meta.output_name()
                        gdf[name] = pl_df.select(t.alias(name)).get_column(name).to_list()
                        tooltip_cols.append(name)
                    else:
                        if t not in gdf.columns:
                            raise ValueError(f"Tooltip column '{t}' not found")
                        tooltip_cols.append(t)
            else:
                if isinstance(tooltip, pl.Expr):
                    name = tooltip.meta.output_name()
                    gdf[name] = pl_df.select(tooltip.alias(name)).get_column(name).to_list()
                    tooltip_cols = [name]
                else:
                    if tooltip not in gdf.columns:
                        raise ValueError(f"Tooltip column '{tooltip}' not found")
                    tooltip_cols = tooltip

        if col_name is None:
            sk = dict(style_kwds)
            sk.setdefault("style_function", lambda feat, c=default_color: {"color": c, "fillColor": c})
            return gdf.explore(
                column=None,
                tooltip=tooltip_cols,
                tiles=tiles,
                cmap=cmap,
                style_kwds=sk,
                **kwargs,
            )

        if color_identity:
            sk = dict(style_kwds)

            radius = sk.get("radius", 10)
            fill_opacity = sk.get("fillOpacity", 0.8)
            weight = sk.get("weight", 1)

            user_style_fn = sk.get("style_function", None)

            def _identity_style(feat, cn=col_name):
                props = feat.get("properties", {})
                c = props.get(cn, default_color)
                base = {
                    "fillColor": c,
                    "color": c,
                    "radius": radius,
                    "fillOpacity": fill_opacity,
                    "weight": weight,
                }
                if callable(user_style_fn):
                    out = user_style_fn(feat)
                    if isinstance(out, dict):
                        base.update(out)
                return base

            sk["style_function"] = _identity_style

            return gdf.explore(
                column=None,
                tooltip=tooltip_cols,
                tiles=tiles,
                cmap=cmap,
                style_kwds=sk,
                **kwargs,
            )

        return gdf.explore(
            column=col_name,
            tooltip=tooltip_cols,
            tiles=tiles,
            cmap=cmap,
            style_kwds=style_kwds,
            **kwargs,
        )


# transformers models

class ViTEmbedder:
    def __init__(self, model_name="google/vit-base-patch16-224", pooling="mean"):
        from transformers import AutoImageProcessor, ViTModel
        import torch 

        self.pooling = pooling.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=False).eval().to(self.device)

    def __call__(self, image_path):
        import torch
        from PIL import Image

        with torch.inference_mode():
            inputs = {k: v.to(self.device) for k, v in self.processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt").items()}
            hidden = self.model(**inputs).last_hidden_state
            vec = torch.nn.functional.normalize(hidden[:, 0, :] if self.pooling == "cls" else hidden[:, 1:, :].mean(dim=1), p=2, dim=-1)
            return vec.squeeze(0).cpu().numpy()


class SigLIPEmbedder:
    def __init__(self, model_name="google/siglip-base-patch16-256", device=None):
        from transformers import AutoProcessor, SiglipModel
        import torch 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = SiglipModel.from_pretrained(model_name).eval().to(self.device)

    def embed_image(self, image_path):
        import torch
        from PIL import Image

        with torch.inference_mode():

            inputs = self.processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vec = torch.nn.functional.normalize(self.model.get_image_features(**inputs), p=2, dim=-1)
            return vec.squeeze(0).cpu().numpy()

    def embed_text(self, text):
        import torch

        with torch.inference_mode():
            inputs = self.processor(text=text, return_tensors="pt", padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vec = torch.nn.functional.normalize(self.model.get_text_features(**inputs), p=2, dim=-1)
            return vec.squeeze(0).cpu().numpy()


class E5TextEmbedder:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-large",
        device=None,
        max_length=512,
        prefix="query: ",
        batch_size=32,
    ):
        from transformers import AutoModel, AutoTokenizer
        import torch 
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.prefix = prefix
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _as_list(self, texts):
        is_single = isinstance(texts, str)
        if is_single:
            return [texts], True

        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a string or a sequence (list/tuple) of strings")

        if len(texts) == 0:
            raise ValueError("texts sequence is empty")

        if not all(isinstance(t, str) for t in texts):
            bad = [type(t).__name__ for t in texts if not isinstance(t, str)]
            raise TypeError(f"All items in texts must be strings. Found non-strings: {bad[:5]}")

        return list(texts), False

    def __call__(self, texts):
        import torch

        with torch.inference_mode():

            texts_list, is_single = self._as_list(texts)

            prefixed = [f"{self.prefix}{t}" for t in texts_list]

            all_vecs = []

            for i in range(0, len(prefixed), self.batch_size):
                batch = prefixed[i : i + self.batch_size]

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

                all_vecs.append(normalized.cpu().numpy())

            vecs = np.vstack(all_vecs)

            return vecs[0] if is_single else vecs


# pytorch

class DSTorch:
    def __new__(cls, *args, **kwargs):
        raise TypeError("DSTorch is a static utility class and cannot be instantiated")

    @staticmethod
    def load_image(df, label_name="label", filepath="filepath", index="index", scale=False, flatten=False):
        import torch
        from PIL import Image

        imgs = [np.array(Image.open(fp)) for fp in df[filepath]]
        X = np.stack(imgs)
        if X.ndim == 3:
            X = X[:, :, :, np.newaxis]
        X = np.transpose(X, (0, 3, 1, 2))
        if scale:
            X = X / 255.0
        if flatten:
            X = X.reshape(X.shape[0], -1)

        y_raw = df[label_name].to_numpy()
        cn = np.unique(y_raw)
        cn_to_idx = {c: i for i, c in enumerate(cn)}
        y = np.array([cn_to_idx[c] for c in y_raw])

        idx = df[index].to_numpy()
        train_mask = idx == "train"
        test_mask = idx == "test"

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        return X, X_train, X_test, y, y_train, y_test, cn

    @staticmethod
    def load_text(df, model, tokens_expr, label_expr, max_length=200, pad_idx=0, unknown_policy="drop", class_order="sorted"):
        import torch
        from sklearn.model_selection import train_test_split

        word_to_idx = model.wv.key_to_index

        if unknown_policy == "drop":
            def tokens_to_indices(tokens):
                idxs = [word_to_idx[t] for t in tokens if t in word_to_idx][:max_length]
                return idxs + [pad_idx] * (max_length - len(idxs))
        else:
            def tokens_to_indices(tokens):
                idxs = [word_to_idx.get(t, pad_idx) for t in tokens][:max_length]
                return idxs + [pad_idx] * (max_length - len(idxs))

        labels = df.select(label_expr.alias("label")).get_column("label").to_list()

        if class_order == "appearance":
            cn = list(dict.fromkeys(labels))
        else:
            cn = sorted(set(labels))

        label_to_idx = {c: i for i, c in enumerate(cn)}

        out = df.select(
            tokens_expr
            .map_elements(tokens_to_indices, return_dtype=pl.List(pl.Int64))
            .alias("indices"),
            label_expr
            .map_elements(lambda x: label_to_idx[x], return_dtype=pl.Int64)
            .alias("y"),
        )

        X = torch.tensor(out.get_column("indices").to_list(), dtype=torch.long)
        y = torch.tensor(out.get_column("y").to_list(), dtype=torch.long)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X, X_train, X_test, y, y_train, y_test, cn

    @staticmethod
    def train(
        model, optimizer, X_train, y_train, num_epochs=18, batch_size=32):
        import torch

        model.train()
        n_samples = len(X_train)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (n_samples // batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    @staticmethod
    def score_image(model, X, y, cn):
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(X)
            pred = logits.argmax(dim=1)
            return (pred == y).float().mean().item()

    @staticmethod
    def score_text(model, X, y):
        import torch

        model.eval()
        with torch.no_grad():
            outputs = model(X)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()
        return accuracy

    @staticmethod
    def predict(model, X, y, cn):
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(X)
        y_idx = y.numpy()
        pred_idx = logits.argmax(dim=1).numpy()
        return pl.DataFrame({
            "target_": [cn[i] for i in y_idx],
            "prediction_": [cn[i] for i in pred_idx]
        })

    @staticmethod
    def predict_proba(model, X, y, cn):
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
        y_idx = y.numpy()
        pred_idx = logits.argmax(dim=1).numpy()
        probs_np = probs.numpy()
        max_probs = probs_np.max(axis=1)
        data = {
            "target_": [cn[i] for i in y_idx],
            "prediction_": [cn[i] for i in pred_idx],
            "prob_pred_": max_probs.tolist()
        }
        for i, name in enumerate(cn):
            data[str(name)] = probs_np[:, i].tolist()
        return pl.DataFrame(data)

    @staticmethod
    def confusion_matrix(model, X, y, cn):
        import torch
        from sklearn.metrics import ConfusionMatrixDisplay

        model.eval()
        with torch.no_grad():
            logits = model(X)
        y_idx = y.numpy()
        pred_idx = logits.argmax(dim=1).numpy()
        y_true = [cn[i] for i in y_idx]
        y_pred = [cn[i] for i in pred_idx]
        return ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")

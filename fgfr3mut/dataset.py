"""Dataset loader of TCGA."""
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd

from fgfr3mut.utils import load_features_to_mem


class TCGADataset:
    """TCGA-BLCA dataset for FGFR3 mutation prediction."""

    def __init__(self, data_dir: Path, mpp: float = 0.5):
        if mpp == 1.0:
            features_dir = data_dir / "features_mpp10"
        elif mpp == 0.5:
            features_dir = data_dir / "features"
        else:
            raise ValueError("mpp must be either 1.0 or 0.5")
        self.features_paths = list(features_dir.glob("**/features.npy"))

        self.filtered_slides_path_tcga = data_dir / "filtered_slides_tcga.xlsx"
        self.loeffler_tcga = data_dir / "loeffler_tcga.xlsx"
        self.target_tcga = data_dir / "mutations_blca_tcga_pancancer_atlas_cbioportal.txt"

    def get_ids_to_keep(
        self,
        cases: List[Literal["MIBC", "NMIBC"]] = ["MIBC"],
    ) -> List[str]:
        """Keep only specific ids, corresponding for instance to MIBC slides."""
        mibc_df = pd.read_excel(self.filtered_slides_path_tcga, index_col=0)
        # To make sure that if we specify NMIBC, both NMIBC - pT1 and NMIBC - pTa are kept
        mibc_df["DX_REV_Summary"].replace(
            {"NMIBC - pT1": "NMIBC", "NMIBC - pTa": "NMIBC"}, inplace=True
        )
        mibc_df = mibc_df[mibc_df["DX_REV_Summary"].isin(cases)]
        return mibc_df.index.tolist()

    def load_fgfr3_status(
        self,
        binarize: bool = True,
        keep_cases: List[Literal["MIBC", "NMIBC"]] = ["MIBC"],
        loeffler_cases: bool = False,
    ) -> pd.Series:
        """Load FGFR3 mutation status (only MIBC tumors).

        Parameters
        ----------
        binarize: bool
            Whether to return 0/1 labels or the real classes names.
        keep_cases: List[Literal["MIBC", "NMIBC"]]  # noqa
            Which cases to keep
        loeffler_cases: bool
            Whether to use cases from Loeffler et al. 2021. If True, overrides `keep_cases`.

        Returns
        -------
        labels : pd.Series
            Ground truth labels.
        """
        if loeffler_cases:
            print("Loading cases and labels from Loeffler et al. 2021")
            # Load supplementary data of
            # https://www.eu-focus.europeanurology.com/article/S2405-4569(21)00113-9/\
            # fulltext
            df = pd.read_excel(self.loeffler_tcga, index_col=1)
            df = df[~df.index.isna()]  # there is one last row that should be removed
            df.rename(
                columns={"molecular FGFR3 mutational statusa": "fgfr3_mutation"},
                inplace=True,
            )
            df["fgfr3_mutation"].replace({"wt": "WT", "mut": "MUT"}, inplace=True)
        else:
            # Mutation
            # Table Mutations (OQL is not in effect)
            # from "https://www.cbioportal.org/results/download?cancer_study_list="
            # "blca_tcga_pan_can_atlas_2018&tab_index=tab_visualize&case_set_id="
            # "blca_tcga_pan_can_atlas_2018_all&Action=Submit&gene_list=FGFR3"
            df = pd.read_csv(self.target_tcga, sep="\t")
            df.index = df["SAMPLE_ID"].map(lambda s: s[:12]).values
            df.rename(columns={"FGFR3": "fgfr3_mutation"}, inplace=True)

            # Change label of non-activated mutations as advised by Markus
            non_activated_fgfr3_patients = [
                "TCGA-XF-A9SL",
                "TCGA-UY-A78N",
                "TCGA-FJ-A3ZF",
                "TCGA-XF-A9SJ",
                "TCGA-4Z-AA81",
                "TCGA-DK-A3IS",
            ]
            print(
                f"Putting to WT the labels of {non_activated_fgfr3_patients} cases that"
                " have non activated FGFR3 mutations/fusions"
            )
            df.loc[non_activated_fgfr3_patients, "fgfr3_mutation"] = "WT"

            if len(keep_cases):
                ids_to_keep = self.get_ids_to_keep(cases=keep_cases)
                common_ids = list(set(df.index).intersection(ids_to_keep))
                print(
                    f"Keeping only {keep_cases} cases: {len(set(df.index))} -> {len(common_ids)}"
                )
                df = df.loc[common_ids]

        labels = df["fgfr3_mutation"]
        labels = labels[~labels.isna()]
        if binarize:
            labels = (labels != "WT").astype(float)
        assert not labels.index.duplicated().any()
        print(f"fgfr3_mutation loaded: {labels.value_counts()}")
        return labels

    def get_features(
            self,
            n_tiles: int,
            num_workers: int,
            features_as: Literal["array", "list", "path"],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the features of every tile for every slide."""
        if features_as in ["array", "list"]:
            X, X_coords, X_slidenames, X_ids = load_features_to_mem(
                features_paths=self.features_paths,
                extract_slidename=self._extract_slidename,
                extract_id=self._extract_slideid,
                n_tiles=n_tiles,
                num_workers=num_workers,
                as_list=features_as == "list",
            )
        else:
            assert features_as == "path"

            X = self.features_paths
            X_coords = None
            X_slidenames = np.squeeze([self._extract_slidename(p) for p in self.features_paths])
            X_ids = np.squeeze(
                [self._extract_slideid(self._extract_slidename(p)) for p in self.features_paths]
            )

        return X, X_coords, X_slidenames, X_ids

    @staticmethod
    def _extract_slidename(path: Path) -> str:
        return path.parents[0].name

    @staticmethod
    def _extract_slideid(slide_name: str) -> str:
        return slide_name[:12]

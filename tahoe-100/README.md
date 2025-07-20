Tahoe-100
=========

# Overview

* **Format:** 
  * Count matrices: h5ad (AnnData)
  * Metadata: Parquet
* **Data host:** 
  * Google Cloud Storage
  * Path: `gs://arc-ctc-tahoe100/`
* **Statistics**
  * Sample count: 1344
  * Cell count: 100648790

## Manuscript

[Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas for Context-Dependent Gene Function and Cellular Modeling](https://doi.org/10.1101/2025.02.20.639398)

## obs (cell) metadata

Here's the table formatted with consistent spacing:

| Column Name            | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| **plate**              | Plate identifier                                                       |
| **BARCODE_SUB_LIB_ID** | Cell identifier                                                        |
| **sample**             | Unique treatment identifier, distinguishes replicated treatments       |
| **gene_count**         | Number of genes with at least one count                                |
| **tscp_count**         | Number of transcripts, aka UMI count                                   |
| **mread_count**        | Number of reads per cell                                               |
| **drugname_drugconc**  | Drug name, concentration, and concentration unit                       |
| **drug**               | Drug name, parsed out from the `drugname_drugconc` field               |
| **cell_line**          | Cell line Cellosaurus identifier                                       |
| **sublibrary**         | Sublibrary ID (related to library prep and sequencing)                 |
| **BARCODE**            | Barcode ID                                                             |
| **pcnt_mito**          | Percentage of mitochondrial reads                                      |
| **S_score**            | Inferred S phase score                                                 |
| **G2M_score**          | Inferred G2M score                                                     |
| **phase**              | Inferred cell cycle phase                                              |
| **pass_filter**        | "Full" filters are more stringent on `gene_count` and `tscp_count`     |
| **cell_name**          | Commonly-used cell name (related to the `cell_line` field)             |


# Tutorials

* [Python](./tutorial-py.ipynb)

# Notes

* `.h5ad.gz` file extensions denote internal gzip compression. 
  * See the [Python tutorial](./tutorial-py.ipynb) on reading in the anndata objects.


# Resources

* [Vevo Theraputics](https://www.vevo.ai/)


# Contact

* [GitHub Issues](https://github.com/ArcInstitute/arc-virtual-cell-atlas/issues)
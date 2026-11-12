# common/references/

Reference literature for all topics. Only this index (`README.md`) and the BibTeX database (`references.bib`) are tracked by git. The PDF files are kept locally and are **not committed**, because their licenses and redistribution terms are not controlled by this repository.

## Policy

- Prefer open-access papers and books. Record the access status and license in the index below.
- Each reference is reachable as `<key>.pdf`, where `<key>` is its BibTeX key in `references.bib`. That name is a **symbolic link** to the original PDF, so the original file keeps its name. Create the link with `ln -sn <original>.pdf <key>.pdf` from this directory. The `File` column gives the original file name.
- Every entry in `references.bib` has an index row here. If a document is only available as a web page, record the URL in `Source` and write `not stored locally` in `File`.
- Do not commit PDFs. Stage only `common/references/README.md` and `common/references/references.bib` from this directory. `.gitignore` excludes every other file here.
- Cite a reference in any document or notebook where its content, equations, figures, or results are used. Paraphrase and cite; do not copy long passages.

## Index

| Key | File | Title | Authors | Year | Source | Access and license | Used in |
|---|---|---|---|---|---|---|---|
| `chao2025` | `2025-lectures-on-accelerator-physics.pdf` (link `chao2025.pdf`) | Lectures on Accelerator Physics, 2nd ed. (1101 pages) | Alexander Wu Chao | 2025 | DOI [10.1142/14320](https://doi.org/10.1142/14320) (local copy) | Commercial book, not open access. Local copy only. | (none yet) |
| `wiedemann2015` | `HWiedemannParticleAcceleratorPhysics.pdf` (link `wiedemann2015.pdf`) | Particle Accelerator Physics, 4th ed. (1029 pages) | Helmut Wiedemann | 2015 | DOI [10.1007/978-3-319-18317-6](https://doi.org/10.1007/978-3-319-18317-6) (local copy) | Commercial book, not open access. Local copy only. | (none yet) |
| `lee2019` | `SYLeeAcceleratorPhysics4th.pdf` (link `lee2019.pdf`) | Accelerator Physics, 4th ed. (569 pages) | S. Y. Lee | 2019 | DOI [10.1142/11111](https://doi.org/10.1142/11111) (local copy) | Commercial book, not open access. Local copy only. | (none yet) |
| `sagan1999` | `PhysRevSTAB.2.074001.pdf` (link `sagan1999.pdf`) | Linear analysis of coupled lattices | David Sagan, David Rubin | 1999 | DOI [10.1103/PhysRevSTAB.2.074001](https://doi.org/10.1103/PhysRevSTAB.2.074001) (local copy) | Phys. Rev. ST Accel. Beams. Publisher access terms not checked. | (none yet) |
| `sagan2000erratum` | `PhysRevSTAB.3.059901.pdf` (link `sagan2000erratum.pdf`) | Erratum: Linear analysis of coupled lattices | David Sagan, David Rubin | 2000 | DOI [10.1103/PhysRevSTAB.3.059901](https://doi.org/10.1103/PhysRevSTAB.3.059901) (local copy) | Erratum to `sagan1999`. Publisher access terms not checked. | (none yet) |
| `soutome2024` | `Skew-Q とカップリング.pdf` (link `soutome2024.pdf`) | Skew-Q とカップリング (Skew quadrupoles and coupling), internal slides, 27 pages | Koichi Soutome (romanization to verify) | 2024 | internal note, no DOI | Internal note in Japanese. Not published. Do not redistribute. | (none yet) |
| `takao_coupling` | `coupling.pdf` (link `takao_coupling.pdf`) | Impact of Betatron Coupling, presentation slides, 19 pages | M. Takao | unknown | presentation, no DOI | Year, venue, and access status unknown. Provisional key. | (none yet) |

The DOIs for the three books were supplied by the author. The DOIs for the two Physical Review ST Accel. Beams entries were taken from the citation printed on the first page of each PDF. All DOIs resolve to the publisher pages. The edition and year of the books were partly inferred from the PDF metadata, so verify them before citing.

Open items: the year of `takao_coupling` and its provisional key, the romanized name in `soutome2024`, and the access terms of the non-book entries.

## Adding a reference

1. Add the BibTeX entry to `references.bib` with a key of the form `firstauthorYEAR` (lowercase ASCII), for example `wiedemann2015`.
2. If you have the PDF, store the original file in this directory and create the link `common/references/<key>.pdf` as a symbolic link to it.
3. Add a row to the index above with the source, the access status, and the license note.
4. Update the `Used in` column when a topic cites the reference.

## License

This index and `references.bib` are documents and are licensed under [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). The referenced PDFs are third-party works, are not part of this repository, and remain under their own licenses. See [LICENSING.md](../../LICENSING.md).

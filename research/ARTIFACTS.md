# Local historical artifact manifest

Large arrays and checkpoints are intentionally excluded from Git. The
historical SigLIP2-B ablations documented in `experiments/` currently require:

| Local path | Size | SHA-256 |
|---|---:|---|
| `archive/data/image_embeddings_siglip2.npy` | 76 MB | `ca6fa38330411930c995a714c649ca45be8c76524c13013e58f041424f66f977` |
| `archive/data/caption_embeddings_siglip2.npy` | 76 MB | `f2008cd0dc3d34aa7fca5da157b4c40b535c1249b179df3fbdcf74b0cc794612` |
| `archive/data/concept_directions_siglip2_final.npy` | 772 KB | `6f8e876c28d2b01b81a583dc3ef2a0927cc94e9b1f54513f494e519c8ab8bf45` |
| `archive/data/image_embeddings_cc0_siglip2.npy` | 5.4 MB | `6d35e09cfeb6b5f35417756edbd39b7b511f36604210580b7e2541bff27e5b30` |
| `outputs/evals/labels_siglip2_final_reconstructed.parquet` | 38 MB | `45baa70d2fc247d58aa8cce1c5ea355587872d70aa5135d878401aa54e11203e` |

These files belong to historical SigLIP2-B/PE experiments and are not inputs to
the maintained SigLIP2 Giant CUDA pipeline. The manifest prevents local
historical artifacts from silently changing while they remain outside version
control. Before publication, any artifact needed for a reported historical
result should be placed in a versioned external store and cited by release
identifier.

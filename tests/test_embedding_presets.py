"""Model selection, input validation and exported-weight compatibility checks."""
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import torch
import rpcontact_cli as cli
from training.rpcontact_training.model import ProteinRNAInteraction_rela

ROOT = Path(__file__).resolve().parents[1]


class EmbeddingPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_existing_default_is_retained(self):
        args = cli.build_parser().parse_args(['-i', 'input.zip'])
        self.assertEqual(args.model, 'auto')
        protein, rna, _, _, mode, _ = cli.load_inputs(ROOT / 'examples/rpcontact_example_input.zip')
        self.assertEqual(mode, 'embedding')
        self.assertEqual((protein.shape[1], rna.shape[1]), (5140, 772))
        self.assertEqual(cli.select_embedding_variant(protein.shape[1], rna.shape[1]), 'rpcontact')
        explicit = cli.load_inputs(ROOT / 'examples/rpcontact_example_input.zip', model_variant='rpcontact')
        np.testing.assert_array_equal(protein, explicit[0])
        np.testing.assert_array_equal(rna, explicit[1])

    def test_resm_zip_and_directory_feature_layouts(self):
        archive = ROOT / 'examples/rpcontact_resm_esmc6b_input.zip'
        protein, rna, pro_seq, rna_seq, mode, _ = cli.load_inputs(archive)
        self.assertEqual(mode, 'embedding')
        self.assertEqual((protein.shape, rna.shape), ((56, 2580), (41, 1284)))
        self.assertEqual(cli.select_embedding_variant(protein.shape[1], rna.shape[1]), 'resm-esmc6b')
        np.testing.assert_array_equal(protein[:, :20], cli.sequence_to_onehot(pro_seq, 'ACDEFGHIKLMNPQRSTVWY'))
        np.testing.assert_array_equal(rna[:, :4], cli.sequence_to_onehot(rna_seq, 'ACGU'))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with zipfile.ZipFile(archive) as z:
                z.extractall(directory)
            for layout in ['raw', 'onehot-first', 'onehot-last']:
                if layout == 'onehot-first':
                    np.save(directory / 'pro.npy', protein); np.save(directory / 'rna.npy', rna)
                elif layout == 'onehot-last':
                    np.save(directory / 'pro.npy', np.concatenate([protein[:, 20:], protein[:, :20]], axis=1))
                    np.save(directory / 'rna.npy', np.concatenate([rna[:, 4:], rna[:, :4]], axis=1))
                for variant in ['auto', 'resm-esmc6b']:
                    actual = cli.load_inputs(directory, model_variant=variant)
                    with self.subTest(layout=layout, model=variant):
                        self.assertEqual(actual[4], 'embedding')
                        np.testing.assert_array_equal(actual[0], protein)
                        np.testing.assert_array_equal(actual[1], rna)

    def test_auto_rejects_unmatched_or_incomplete_embeddings(self):
        for protein_dim, rna_dim in [(5120, 1280), (2560, 768), (100, 100)]:
            with self.subTest(dimensions=(protein_dim, rna_dim)):
                with self.assertRaisesRegex(ValueError, 'No unique bundled checkpoint'):
                    cli.select_embedding_variant(protein_dim, rna_dim)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pro.fasta').write_text('>protein\nMTTU\n')
            (root / 'rna.fasta').write_text('>rna\nACTT\n')
            np.save(root / 'pro.npy', np.zeros((4, 2560), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'together'):
                cli.load_inputs(root)
            np.save(root / 'rna.npy', np.zeros((4, 768), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'No unique bundled checkpoint'):
                cli.load_inputs(root)

    def test_resm_request_rejects_wrong_or_missing_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'requires matching'):
                cli.predict(ROOT / 'examples/rpcontact_example_input.zip', root / 'wrong', model_variant='resm-esmc6b')
            fasta = root / 'fasta'; fasta.mkdir()
            (fasta / 'pro.fasta').write_text('>protein\nMTTU\n')
            (fasta / 'rna.fasta').write_text('>rna\nACTT\n')
            with self.assertRaisesRegex(ValueError, 'requires matching'):
                cli.predict(fasta, root / 'missing', model_variant='resm-esmc6b')
            self.assertFalse((root / 'wrong/rpcontact_summary.json').exists())
            self.assertFalse((root / 'missing/rpcontact_summary.json').exists())
            # Existing FASTA-only mode still preserves protein T/U and normalizes RNA.
            protein, rna, pro_seq, rna_seq, mode, _ = cli.load_inputs(fasta)
            self.assertEqual((pro_seq, rna_seq, mode), ('MTTU', 'ACUU', 'one-hot'))
            self.assertEqual(protein[1, 16], 1)
            self.assertEqual(protein[3].sum(), 0)

    def test_exported_weights_match_training_architecture_and_score_chunks(self):
        checkpoint = ROOT / 'models/rpcontact_resm_esmc6b_state_dict.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        reference = ProteinRNAInteraction_rela(2580, 1284, 3, 48, n_head=4, d_ff=512, dropout=.1, sync=True)
        reference.load_state_dict(state, strict=True); reference.eval()
        protein, rna, *_ = cli.load_inputs(ROOT / 'examples/rpcontact_resm_esmc6b_input.zip', model_variant='resm-esmc6b')
        with torch.no_grad():
            expected = reference(torch.from_numpy(protein.T[None]), torch.from_numpy(rna.T[None]))[0, :, :, 0].numpy()
        deployed = cli.load_checkpoint_model(2580, 1284, checkpoint)
        deployed.load_state_dict(state, strict=True)
        f_pro, f_rna = cli.encode_tracks(deployed, protein, rna)
        actual = np.empty_like(expected)
        for p in range(0, len(protein), 20):
            for r in range(0, len(rna), 15):
                actual[p:p+20, r:r+15] = cli.score_block(deployed, f_pro, f_rna, p, min(p+20, len(protein)), r, min(r+15, len(rna)))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

"""Pure unit checks for the new transport/inventory layer, no native execution."""
import gzip,hashlib,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
import native_host_join as join
import replay

class ExactTransport(unittest.TestCase):
    def test_binary_ties_round_to_even(self):
        for p in (53,113):
            u=F(2)**(1-p)
            self.assertEqual(join.rounded(1+u/2,p),1)
            self.assertEqual(join.rounded(1+3*u/2,p),1+2*u)
            self.assertEqual(join.rounded(-1-u/2,p),-1)
            self.assertEqual(join.rounded(-1-3*u/2,p),-1-2*u)
            tiny=F(2)**(-1074 if p==53 else -16494)
            self.assertEqual(join.rounded(tiny/2,p),0)
            self.assertEqual(join.rounded(3*tiny/2,p),2*tiny)
    def test_complete_equal_dp_qp_mapping_required(self):
        row='      DATA(TIRUST_MAP(I),I=0,209)/'+','.join(map(str,range(1,211)))+'/'
        self.assertEqual(join.mapping(row+'\n'+row),list(range(210)))
        with self.assertRaises(ValueError):join.mapping(row)
        with self.assertRaises(ValueError):join.mapping(row+'\n'+row.replace(',210/',',209/'))
    def test_output_length_rejected(self):
        for kind in (1,2):
            with self.assertRaises(ValueError):join.outputs({'kind':kind,'out':'00'})

class RetainedPayload(unittest.TestCase):
    def test_escape_rejected(self):
        for path in ('../external','a/../../external','/tmp/external'):
            with self.assertRaises(ValueError):replay.safe(path)
    def test_gzip_roundtrip_and_decoded_hash(self):
        old=replay.HERE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                replay.HERE=Path(tmp);raw=b'sealed text\n';stored=gzip.compress(raw,mtime=0)
                (replay.HERE/'record.txt.gz').write_bytes(stored)
                meta={**replay.identity(raw),'stored':'record.txt.gz','stored_identity':replay.identity(stored),'encoding':'gzip'}
                self.assertEqual(replay.retained(meta),raw)
                meta['sha256']='0'*64
                with self.assertRaises(ValueError):replay.retained(meta)
        finally:replay.HERE=old

if __name__=='__main__':unittest.main()

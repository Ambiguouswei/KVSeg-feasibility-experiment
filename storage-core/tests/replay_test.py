import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

BINARY=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[:1]

class ReplayTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ['KVCOHORT_TEST_ROOT']);parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='replay-',dir=parent);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def run_trace(self,text,layout='append',*options):
        path=self.root/f'{layout}.trace';path.write_text(text,encoding='utf-8')
        return subprocess.run([str(BINARY),str(path),str(self.root/layout),layout,*options],capture_output=True,text=True,timeout=30)
    def test_matched_inputs_expose_read_locality(self):
        trace='kvfs-trace-v1\nput 1 0 1 long 1024\nput 2 1 1 short 1024\nread 2\nretire 2\ngc 8\nread 2\n'
        rows=[]
        for layout in ['append','prefix','lifetime','joint']:
            run=self.run_trace(trace,layout);self.assertEqual(run.returncode,0,run.stderr);rows.append(json.loads(run.stdout))
        self.assertEqual([r['usable_prefix_tokens'] for r in rows],[96]*4)
        self.assertEqual([r['requested_tokens'] for r in rows],[128]*4)
        self.assertEqual([r['demand_pread_calls'] for r in rows],[2,2,3,3])
        self.assertEqual([r['gc_moved_bytes'] for r in rows],[1024,1024,0,0])
    def test_invalid_trace_is_rejected_before_creating_store(self):
        for trace in ['kvfs-trace-v1\nput 1 2 1 short 1024\n','kvfs-trace-v1\nput 1 0 1 long 1024\nput 1 0 1 short 1024\n',
                      'kvfs-trace-v1\nput 1 0 1 long 1024\nput 2 0 1 short 1024\n']:
            run=self.run_trace(trace);self.assertNotEqual(run.returncode,0);self.assertFalse((self.root/'append').exists())
    def test_pressure_records_admission_failure_and_later_recovery(self):
        trace='kvfs-trace-v1\nput 1 0 1 long 4096\nput 2 1 1 short 4096\nput 3 1 1 short 4096\nread 3\nread 2\nretire 2\ngc 8\nput 3 1 1 short 4096\nread 3\n'
        run=self.run_trace(trace,'joint','--segment-kib','4','--payload-budget-kib','12')
        self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
        self.assertEqual(row['admission_rejections'],1);self.assertEqual(row['new_admissions'],3)
        self.assertEqual(row['full_prefix_hits'],2);self.assertEqual(row['usable_prefix_tokens'],160)
        self.assertLessEqual(row['max_observed_payload_occupied_bytes'],12*1024)
    def test_replay_refuses_to_reuse_nonempty_root(self):
        text='kvfs-trace-v1\nput 1 0 1 long 1024\n'
        self.assertEqual(self.run_trace(text).returncode,0)
        run=self.run_trace(text);self.assertNotEqual(run.returncode,0);self.assertIn('fresh empty',run.stderr)

    def test_admission_requires_reuse_evidence(self):
        # 根块无证据也可写入；子块在其父前缀被读过之前应被拒绝（declined，而非 capacity 拒绝）
        trace=('kvfs-trace-v1\nput 1 0 1 long 1024\nput 2 1 1 short 1024\nread 2\n'
               'put 2 1 1 short 1024\n')
        run=self.run_trace(trace,'append','--admit-min-reuse','1')
        self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
        self.assertEqual(row['admission_declined'],1)
        self.assertEqual(row['new_admissions'],2)
        self.assertEqual(row['final_objects'],2)
    def test_gc_score_is_reported(self):
        trace=('kvfs-trace-v1\nput 1 0 1 long 1024\nput 2 1 1 short 1024\n'
               'read 2\nretire 2\ngc 8\n')
        for layout,score in [('append','max-dead'),('lifetime','min-cost')]:
            run=self.run_trace(trace,layout,'--gc-score',score)
            self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
            self.assertEqual(row['gc_score'],score)
            self.assertEqual(row['final_objects'],1)

    def test_direct_io_handles_unaligned_payloads(self):
        # 载荷刻意不按 4096 对齐，用来验证 O_DIRECT 的对齐缓冲与补齐逻辑。
        trace=('kvfs-trace-v1\nput 1 0 1 long 5000\nput 2 1 1 short 3000\n'
               'read 2\ngc 8\nread 2\n')
        run=self.run_trace(trace,'joint','--direct-io','1','--segment-kib','64')
        self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
        self.assertEqual(row['requested_tokens'],128)      # 两次 read 2，每次 2 块 x 32 token
        self.assertEqual(row['usable_prefix_tokens'],128)   # 全部命中，且逐块做确定性校验
        self.assertEqual(row['final_objects'],2)

if __name__=='__main__':unittest.main(verbosity=2)

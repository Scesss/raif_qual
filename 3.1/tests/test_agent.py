import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from agent_tools import Documents, ReviewTools, finite_sets, truth_table
from model_gateway import Credentials, GatewayError, ModelGateway, atomic_json, parse_sse
from review_agent import RunManager
from review_storage import ConflictError


class CourseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_patch = patch.object(app, 'DATA', self.root)
        self.data_patch.start()
        self.demo_patch = patch.dict(os.environ, {'REVIEWER_DEMO': '1'})
        self.demo_patch.start()
        self.store = app.Store(self.root / 'review.sqlite3')
        import pymupdf
        self.imports = self.root / 'imports'
        self.imports.mkdir()
        for path, text in [(self.root/'statement.pdf', 'P implies Q iff not P or Q. Find intersection.'),
                           (self.root/'rubric.pdf', '4 points per correct justified answer. 0 for wrong answer.'),
                           (self.imports/'Test Student.pdf', 'CASE: CORRECT. Correct implication proof. Intersection is 2.')]:
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((50, 80), text)
                pdf.save(path)
        self.student = self.store.create_student({'full_name': 'Test Student', 'email': 'test@example.test'})
        self.hw = self.store.create_homework({'number': 1, 'title': 'Demo', 'statement_path': str(self.root/'statement.pdf'),
            'rubric_path': str(self.root/'rubric.pdf'), 'problems': [{'problem': 1, 'subparts': []}, {'problem': 2, 'subparts': []}]})
        self.store.import_files(self.hw['id'], str(self.imports))
        self.review = self.store.list_reviews(self.hw['id'])[0]
        self.gateway = ModelGateway(self.root/'config.json', Credentials(self.root/'auth'))
        self.gateway.configure({'provider': 'replay', 'model': ''})
        self.manager = RunManager(self.store, self.gateway, start_worker=False)
        self.draft = {'scores': [{'problem': n, 'score': 4, 'reason': 'Correct, page 1.', 'page': 1, 'subparts': []} for n in (1, 2)],
                      'evidence': [{'source': 'work', 'page': 1, 'note': 'Both answers justified.'}],
                      'student_comment': 'Correct.', 'flags': [], 'defense_questions': []}

    def tearDown(self):
        self.manager.close()
        self.demo_patch.stop()
        self.data_patch.stop()
        self.tmp.cleanup()

    def run_one(self):
        run = self.manager.start([self.review['id']])[0]
        self.manager.execute(run['id'])
        return self.manager.get(run['id'])

    def test_real_documents_tool_loop_saves_complete_versioned_draft(self):
        run = self.run_one()
        self.assertEqual(run['status'], 'completed', run['error'])
        saved = self.store.review(self.review['id'])
        self.assertEqual(saved['suggested_total'], 8)
        self.assertEqual(len(self.store.review_versions(saved['id'])), 2)
        self.assertEqual(json.loads(saved['provenance_json'])['sources']['work']['sha256'], self.store.source_manifest(saved['id'])['sources']['work']['sha256'])
        names = [e['data'].get('tool') for e in run['events'] if e['kind'] == 'tool_call']
        self.assertIn('check_logic', names)
        self.assertIn('calculate_sets', names)
        self.assertNotIn('data:image', json.dumps(run))

    def test_counterexample_changes_following_tool_choice(self):
        import pymupdf
        work, _ = self.store.submission_file(self.review['submission_id'])
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 80), 'CASE: ERROR. P and Q equals not P or Q.')
            replacement = work.with_suffix('.new.pdf')
            pdf.save(replacement)
        replacement.replace(work)
        run = self.run_one()
        self.assertEqual(run['status'], 'completed', run['error'])
        calls = [e['data'] for e in run['events'] if e['kind'] == 'tool_call']
        check_index = next(i for i,c in enumerate(calls) if c['tool'] == 'check_logic')
        self.assertEqual(calls[check_index+1]['tool'], 'read_page')
        self.assertIn('crop', calls[check_index+1]['arguments'])
        self.assertEqual(self.store.review(self.review['id'])['suggested_total'], 4)

    def test_unclear_work_finishes_without_zero_or_draft(self):
        import pymupdf
        work, _ = self.store.submission_file(self.review['submission_id'])
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 80), 'CASE: UNCLEAR. Operator missing.')
            replacement = work.with_suffix('.new.pdf')
            pdf.save(replacement)
        replacement.replace(work)
        run = self.run_one()
        self.assertEqual(run['status'], 'needs_manual_reading', run['error'])
        self.assertIsNone(self.store.review(self.review['id'])['suggested_total'])

    def test_strict_scores_evidence_and_task_coverage(self):
        for value in (3.9, 3.0, True, '3.9', '3e0', None):
            draft = copy.deepcopy(self.draft)
            draft['scores'][0]['score'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.save_draft(self.review['id'], draft)
        for change in ('missing_task', 'no_reason', 'no_evidence', 'bad_page'):
            draft = copy.deepcopy(self.draft)
            if change == 'missing_task': draft['scores'].pop()
            if change == 'no_reason': draft['scores'][0]['reason'] = ''
            if change == 'no_evidence': draft['evidence'] = []
            if change == 'bad_page': draft['scores'][0]['page'] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.store.save_draft(self.review['id'], draft)

    def test_all_subparts_are_required(self):
        self.store.update_homework_spec(self.hw['id'], {'problems': [{'problem': 1, 'subparts': ['a','b']}, {'problem': 2, 'subparts': []}]})
        with self.assertRaises(ValueError):
            self.store.save_draft(self.review['id'], self.draft)
        draft = copy.deepcopy(self.draft)
        draft['scores'][0]['subparts'] = [{'id': x, 'reason': 'Justified.'} for x in ('a','b')]
        self.assertEqual(self.store.save_draft(self.review['id'], draft)['suggested_total'], 8)

    def test_ten_point_scale_roundtrips_and_rejects_out_of_range_scores(self):
        self.store.update_homework(self.hw['id'], {**self.hw, 'problems': [
            {'problem': n, 'subparts': [], 'max_score': 10} for n in (1, 2)]})
        snapshot = self.store.source_manifest(self.review['id'])
        tools = ReviewTools(self.store, self.review['id'], snapshot)
        context, _ = tools.execute('read_context', {})
        self.assertEqual([p['max_score'] for p in context['problems']], [10, 10])
        for source in ('work', 'statement', 'rubric'):
            tools.execute('read_page', {'source': source, 'page': 1})
        draft = copy.deepcopy(self.draft)
        draft['scores'][0]['score'], draft['scores'][1]['score'] = 10, 7
        result, _ = tools.execute('submit_draft', {'draft': draft})
        self.assertEqual(result['total'], 17)
        saved = self.store.save_draft(self.review['id'], draft)
        self.assertEqual(saved['suggested_total'], 17)
        self.assertEqual(self.store.approve(self.review['id'], {})['approved_total'], 17)
        draft['scores'][0]['score'] = 11
        with self.assertRaises(ValueError):
            self.store.validate_draft(draft, self.hw['id'])
        for maximum in (0, -1, 3.5, True, 101):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.store.validate_problem_spec([{'problem': 1, 'max_score': maximum}])

    def test_scale_change_invalidates_in_flight_and_saved_results(self):
        snapshot = self.store.source_manifest(self.review['id'])
        self.store.save_draft(self.review['id'], self.draft)
        self.store.update_homework_spec(self.hw['id'], {'problems': [
            {'problem': n, 'subparts': [], 'max_score': 10} for n in (1, 2)]})
        with self.assertRaises(ConflictError):
            self.store.save_draft(self.review['id'], self.draft, provenance=snapshot)
        with self.assertRaises(ConflictError):
            self.store.approve(self.review['id'], {})

    def test_history_compares_percentages_across_different_scales(self):
        self.store.save_draft(self.review['id'], self.draft)
        self.store.approve(self.review['id'], {})
        for number in (2, 3):
            homework = self.store.create_homework({**self.hw, 'number': number, 'problems': [
                {'problem': n, 'subparts': [], 'max_score': 10} for n in (1, 2)]})
            self.store.import_files(homework['id'], str(self.imports))
            review = self.store.list_reviews(homework['id'])[0]
            draft = copy.deepcopy(self.draft)
            for score in draft['scores']:
                score['score'] = 10
            self.store.save_draft(review['id'], draft)
            if number == 2:
                self.store.approve(review['id'], {})
        comparison = self.store.history_check(review['id'], {})
        self.assertFalse(comparison['candidate'])
        self.assertEqual(comparison['current_percent'], 100)
        self.assertEqual(comparison['baseline_percent'], 100)
        self.assertEqual(comparison['jump_points'], 0)

    def test_five_point_subparts_must_sum_to_ten_point_task(self):
        self.store.update_homework_spec(self.hw['id'], {'problems': [
            {'problem': 1, 'subparts': ['a', 'b'], 'max_score': 10, 'subpart_max_scores': {'a': 5, 'b': 5}},
            {'problem': 2, 'subparts': [], 'max_score': 10}]})
        draft = copy.deepcopy(self.draft)
        draft['scores'][0].update(score=8, subparts=[
            {'id': 'a', 'score': 5, 'reason': 'Complete.'}, {'id': 'b', 'score': 3, 'reason': 'Partial.'}])
        self.assertEqual(self.store.save_draft(self.review['id'], draft)['suggested_total'], 12)
        for a, b, total in ((6, 2, 8), (5, 2, 8), (None, 3, 8), (True, 3, 4)):
            bad = copy.deepcopy(draft)
            bad['scores'][0]['score'] = total
            bad['scores'][0]['subparts'][0]['score'] = a
            bad['scores'][0]['subparts'][1]['score'] = b
            with self.subTest(a=a, b=b, total=total), self.assertRaises(ValueError):
                self.store.validate_draft(bad, self.hw['id'])
        with self.assertRaises(ValueError):
            self.store.validate_problem_spec([{'problem': 1, 'subparts': ['a', 'b'], 'max_score': 10,
                                             'subpart_max_scores': {'a': 4, 'b': 5}}])

    def test_manual_edit_blocks_queued_stale_run(self):
        run = self.manager.start([self.review['id']])[0]
        self.store.save_draft(self.review['id'], self.draft)
        self.manager.execute(run['id'])
        result = self.manager.get(run['id'])
        self.assertEqual(result['error']['code'], 'Conflict')
        self.assertEqual(len(self.store.review_versions(self.review['id'])), 2)

    def test_material_change_blocks_save_and_approval(self):
        manifest = self.store.source_manifest(self.review['id'])
        self.store.save_draft(self.review['id'], self.draft)
        self.store.update_homework_spec(self.hw['id'], {'problems': [{'problem': 1, 'subparts': []}]})
        with self.assertRaises(ValueError):
            self.store.approve(self.review['id'], {})
        with self.assertRaises(ValueError):
            self.store.save_draft(self.review['id'], self.draft, provenance=manifest)

    def test_cancel_queued_and_cancel_during_model_prevents_commit(self):
        run = self.manager.start([self.review['id']])[0]
        self.manager.cancel(run['id'])
        self.manager.execute(run['id'])
        self.assertEqual(self.manager.get(run['id'])['status'], 'canceled')
        run = self.manager.start([self.review['id']])[0]
        from demo_replay import ReplayGateway
        original = ReplayGateway.complete
        manager = self.manager
        def cancel_on_call(gateway, *args, **kwargs):
            manager.cancel(run['id'])
            return original(gateway, *args, **kwargs)
        with patch.object(ReplayGateway, 'complete', cancel_on_call):
            self.manager.execute(run['id'])
        self.assertEqual(self.manager.get(run['id'])['status'], 'canceled')
        self.assertIsNone(self.store.review(self.review['id'])['suggested_total'])

    def test_stream_failure_never_saves_and_never_changes_billing(self):
        from demo_replay import ReplayGateway
        run = self.manager.start([self.review['id']])[0]
        with patch.object(ReplayGateway, 'complete', side_effect=GatewayError('CapacityUnavailable', 'Limit')):
            self.manager.execute(run['id'])
        self.assertEqual(self.manager.get(run['id'])['error']['code'], 'CapacityUnavailable')
        self.assertEqual(self.gateway.config()['provider'], 'replay')
        self.assertIsNone(self.store.review(self.review['id'])['suggested_total'])

    def test_duplicate_queue_and_approved_reviews_are_rejected(self):
        self.manager.start([self.review['id']])
        with self.assertRaises(ConflictError):
            self.manager.start([self.review['id']])
        self.store.save_draft(self.review['id'], self.draft)
        self.store.approve(self.review['id'], {})
        with self.assertRaises(ValueError):
            self.manager.start([self.review['id']])

    def test_concurrent_writers_only_one_can_commit_same_revision(self):
        revision = self.store.review(self.review['id'])['revision']
        outcomes = []
        def write():
            try:
                self.store.save_draft(self.review['id'], self.draft, expected_revision=revision)
                outcomes.append('saved')
            except ConflictError:
                outcomes.append('conflict')
        threads = [threading.Thread(target=write) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertCountEqual(outcomes, ['saved', 'conflict'])

    def test_restart_marks_running_as_interrupted_and_keeps_queue(self):
        run = self.manager.start([self.review['id']])[0]
        with self.store.session() as db:
            db.execute("UPDATE agent_runs SET status='running' WHERE id=?", (run['id'],))
        resumed = RunManager(self.store, self.gateway, start_worker=False)
        self.assertEqual(resumed.get(run['id'])['status'], 'interrupted')
        resumed.close()

    def test_tool_cannot_access_arbitrary_path_or_skip_pages(self):
        tools = ReviewTools(self.store, self.review['id'], self.store.source_manifest(self.review['id']))
        with self.assertRaises(ValueError): tools.execute('read_page', {'source': '/etc/passwd', 'page': 1})
        with self.assertRaises(ValueError): tools.execute('submit_draft', {'draft': self.draft})
        with self.assertRaises(ValueError): tools.execute('approve', {})

    def test_http_queue_poll_pages_and_cross_origin_rejection(self):
        import agent_http
        from http.server import ThreadingHTTPServer
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        server=ThreadingHTTPServer(('127.0.0.1',0),app.App)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        base='http://127.0.0.1:'+str(server.server_address[1])
        def request(path,payload=None,origin=None):
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            raw=None if payload is None else json.dumps(payload).encode()
            return urlopen(Request(base+path,data=raw,headers=headers),timeout=5)
        with patch.object(app,'STORE',self.store),patch.object(agent_http,'_runtime',self.manager):
            thread.start()
            try:
                with self.assertRaises(HTTPError) as caught:
                    request('/api/agent/runs',{'review_ids':[self.review['id']]},'https://other.example')
                self.assertEqual(caught.exception.code,403)
                with request('/api/agent/runs',{'review_ids':[self.review['id']]}) as response:
                    self.assertEqual(response.status,202)
                    run=json.load(response)[0]
                with self.assertRaises(HTTPError): request('/api/model/config',{'provider':'subscription'})
                self.manager.execute(run['id'])
                with request('/api/agent/runs/'+run['id']) as response:
                    result=json.load(response)
                    self.assertEqual(result['status'],'completed')
                    cursor=result['cursor']
                with request('/api/agent/runs/'+run['id']+'?after='+str(cursor)) as response:
                    self.assertEqual(json.load(response)['events'],[])
                with request('/api/records/'+self.review['id']+'/pages') as response:
                    self.assertEqual(json.load(response)['pages'],1)
                with request('/api/records/'+self.review['id']+'/page?page=1') as response:
                    self.assertEqual(response.read(8),b'\x89PNG\r\n\x1a\n')
                with request('/api/records/'+self.review['id']+'/versions') as response:
                    self.assertEqual(len(json.load(response)),2)
            finally:
                server.shutdown();server.server_close();thread.join()


class GatewayTests(unittest.TestCase):
    def test_sse_deltas_are_not_success(self):
        events = []
        with self.assertRaises(GatewayError) as caught:
            parse_sse(['data: {"type":"response.output_text.delta","delta":"hello"}', '', 'data: [DONE]', ''], threading.Event(), lambda *x: events.append(x))
        self.assertEqual(caught.exception.code, 'StreamIncomplete')
        self.assertEqual(events[0][0], 'text_delta')

    def test_sse_quota_can_fail_after_text(self):
        lines = ['data: {"type":"response.output_text.delta","delta":"hello"}', '',
                 'data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}', '']
        with self.assertRaises(GatewayError) as caught:
            parse_sse(lines, threading.Event(), lambda *_: None)
        self.assertEqual(caught.exception.code, 'CapacityUnavailable')

    def test_sse_success_and_cancellation(self):
        lines = ['data: {"type":"response.completed","response":{"status":"completed","output":[]}}', '']
        self.assertEqual(parse_sse(lines, threading.Event(), lambda *_: None)['status'], 'completed')
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(GatewayError) as caught: parse_sse(lines, cancel, lambda *_: None)
        self.assertEqual(caught.exception.code, 'Canceled')

    def test_oauth_state_pkce_and_storage_without_plan_grant(self):
        import time
        from urllib.parse import parse_qs, urlparse
        with tempfile.TemporaryDirectory() as directory:
            creds = Credentials(directory)
            first = parse_qs(urlparse(creds.start('http://127.0.0.1:8766/auth/callback')).query)
            second = parse_qs(urlparse(creds.start('http://127.0.0.1:8766/auth/callback')).query)
            self.assertNotEqual(first['state'], second['state'])
            self.assertEqual(first['ext_agent_host_id'], second['ext_agent_host_id'])
            self.assertEqual(first['client_id'], ['dynamic_agent_client'])
            self.assertEqual(first['code_challenge_method'], ['S256'])
            with self.assertRaises(GatewayError): creds.finish({'state':['wrong']})
            with patch.object(creds, '_token_request', return_value={'access_token':'test-secret', 'refresh_token':'test-refresh', 'id_token':'id', 'token_type':'Bearer', 'scope':'openid', 'expires_in':3600}), patch.object(creds, '_claims', return_value={'sub':'user','email':'test@example.test'}):
                self.assertFalse(creds.finish({'state':first['state'],'code':['code'],'client_id':['oaiapp_test']}))
            with self.assertRaises(GatewayError) as caught: creds.access_token()
            self.assertEqual(caught.exception.code, 'ScopeMissing')
            self.assertNotIn('test-secret', json.dumps(creds.public()))
            self.assertEqual(creds.path.stat().st_mode & 0o777, 0o600)
            returning = parse_qs(urlparse(creds.start('http://127.0.0.1:8766/auth/callback', 'oaiapp_test')).query)
            self.assertNotIn('agent_name_hint', returning)
            with self.assertRaises(GatewayError): creds.finish({'state':returning['state'],'code':['code'],'client_id':['different']})

    def test_refresh_is_serialized_and_rotating_token_saved(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            creds = Credentials(directory)
            atomic_json(creds.path, {'active':'client','profiles':{'client':{'subject':'user','access_token':'old','refresh_token':'refresh-old','expires_at':0,'scopes':['chatgpt.tokens.use.direct']}}})
            with patch.object(creds, '_token_request', return_value={'access_token':'new','refresh_token':'refresh-new','token_type':'Bearer','expires_in':3600}) as refresh:
                results=[]
                threads=[threading.Thread(target=lambda:results.append(creds.access_token())) for _ in range(3)]
                for t in threads:t.start()
                for t in threads:t.join()
                self.assertEqual(results,['new']*3)
                refresh.assert_called_once()
            self.assertEqual(creds.read()['profiles']['client']['refresh_token'],'refresh-new')

    def test_paid_api_is_explicit_and_subscription_errors_do_not_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENAI_API_KEY':'paid-secret'}):
            gateway = ModelGateway(Path(directory)/'config.json', Credentials(Path(directory)/'auth'))
            self.assertEqual(gateway.config()['provider'],'subscription')
            with self.assertRaises(GatewayError): gateway.token('subscription')
            gateway.configure({'provider':'api','model':'test-model'})
            self.assertEqual(gateway.token('api'),'paid-secret')
            self.assertFalse(gateway.status()['automatic_paid_fallback'])

    def test_id_token_signature_audience_and_nonce_are_verified(self):
        import time
        import httpx
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
        jwk['kid'] = 'test-signing-key'
        def request(req):
            if req.url.path == '/.well-known/openid-configuration':
                return httpx.Response(200, json={'issuer':'https://auth.openai.com','jwks_uri':'https://auth.openai.com/jwks'})
            return httpx.Response(200, json={'keys':[jwk]})
        factory = lambda: httpx.Client(transport=httpx.MockTransport(request))
        with tempfile.TemporaryDirectory() as directory:
            creds = Credentials(directory, client_factory=factory)
            claims = {'iss':'https://auth.openai.com','sub':'user','aud':'client','exp':time.time()+3600,'nonce':'expected'}
            token = jwt.encode(claims, private, algorithm='RS256', headers={'kid':'test-signing-key'})
            self.assertEqual(creds._claims(token,'client','expected')['sub'],'user')
            for client,nonce in [('wrong','expected'),('client','wrong')]:
                with self.subTest(client=client,nonce=nonce), self.assertRaises(GatewayError):
                    creds._claims(token,client,nonce)
            forged = jwt.encode({**claims,'sub':'other'}, 'not-the-private-key-not-the-private-key', algorithm='HS256', headers={'kid':'test-signing-key'})
            with self.assertRaises(GatewayError): creds._claims(forged,'client','expected')

    def test_http_stream_payload_namespace_and_encrypted_history(self):
        import httpx
        seen=[]
        def request(req):
            seen.append(json.loads(req.content))
            self.assertEqual(req.headers['authorization'],'Bearer paid-secret')
            self.assertEqual(req.extensions['timeout']['read'], 180)
            self.assertEqual(req.extensions['timeout']['write'], 60)
            event={'type':'response.completed','response':{'status':'completed','output':[{'type':'reasoning','id':'r','encrypted_content':'opaque'}]}}
            return httpx.Response(200, text='data: '+json.dumps(event)+'\n\n', headers={'Content-Type':'text/event-stream'})
        factory=lambda:httpx.Client(transport=httpx.MockTransport(request))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}):
            gateway=ModelGateway(Path(directory)/'settings.json',Credentials(Path(directory)/'auth'),client_factory=factory)
            gateway.configure({'provider':'api','model':'test-model'})
            from agent_tools import TOOLS
            result=gateway.complete([{'role':'user','content':'test'}],'instructions',TOOLS,threading.Event(),lambda *_:None)
            self.assertEqual(result['output'][0]['encrypted_content'],'opaque')
            self.assertFalse(seen[0]['store'])
            self.assertTrue(seen[0]['stream'])
            self.assertEqual(seen[0]['tools'][0]['type'],'namespace')
            self.assertIn('reasoning.encrypted_content',seen[0]['include'])
            self.assertNotIn('previous_response_id',seen[0])

    def test_timeout_and_disconnect_retry_same_context_without_partial_tool_results(self):
        import httpx
        seen, events = [], []
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b'data: {"type":"response.output_text.delta","delta":"unfinished"}\n\n'
                raise httpx.ReadTimeout('private transport detail')
        def request(req):
            seen.append(json.loads(req.content))
            if len(seen) == 1:
                return httpx.Response(200, stream=BrokenStream(), headers={'x-request-id':'req-first'})
            if len(seen) == 2:
                raise httpx.RemoteProtocolError('private connection detail')
            result={'status':'completed','output':[{'type':'function_call','name':'read_context','call_id':'once','arguments':'{}'}]}
            return httpx.Response(200, text='data: '+json.dumps({'type':'response.completed','response':result})+'\n\n')
        factory=lambda:httpx.Client(transport=httpx.MockTransport(request))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}), patch('model_gateway.RETRY_BASE_DELAY',0):
            gateway=ModelGateway(Path(directory)/'settings.json',client_factory=factory)
            gateway.configure({'provider':'api','model':'test-model'})
            history=[{'role':'user','content':'test'},{'type':'function_call_output','call_id':'previous','output':'already read page'}]
            result=gateway.complete(history,'instructions',[],threading.Event(),lambda kind,data:events.append((kind,data)))
            self.assertEqual(result['output'][0]['call_id'],'once')
            self.assertEqual(len(seen),3)
            self.assertTrue(all(body==seen[0] for body in seen))
            retries=[data for kind,data in events if kind=='request_retry']
            self.assertEqual([r['code'] for r in retries],['ModelTimeout','StreamDisconnected'])
            self.assertEqual(retries[0]['request_id'],'req-first')
            self.assertEqual(gateway.config()['provider'],'api')
            self.assertNotIn('private',json.dumps(events))
            self.assertNotIn('paid-secret',json.dumps(events))

    def test_retry_is_bounded_and_keeps_timeout_request_id(self):
        import httpx
        attempts=[]
        class SlowStream(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadTimeout('private detail')
                yield b''
        def request(req):
            attempts.append(req)
            return httpx.Response(200,stream=SlowStream(),headers={'x-request-id':'req-slow'})
        factory=lambda:httpx.Client(transport=httpx.MockTransport(request))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}), patch('model_gateway.RETRY_BASE_DELAY',0):
            gateway=ModelGateway(Path(directory)/'settings.json',client_factory=factory)
            gateway.configure({'provider':'api','model':'test-model'})
            with self.assertRaises(GatewayError) as caught:
                gateway.complete([], 'instructions', [], threading.Event(), lambda *_: None)
            self.assertEqual(caught.exception.code,'ModelTimeout')
            self.assertEqual(caught.exception.request_id,'req-slow')
            self.assertEqual(caught.exception.public()['transport_error'],'ReadTimeout')
            self.assertEqual(len(attempts),3)

    def test_connection_failure_has_actionable_message_without_transport_secrets(self):
        import httpx
        def request(req):
            raise httpx.ConnectError('private DNS transport detail')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}), patch('model_gateway.RETRY_BASE_DELAY',0):
            gateway=ModelGateway(Path(directory)/'settings.json',client_factory=lambda:httpx.Client(transport=httpx.MockTransport(request)))
            gateway.configure({'provider':'api','model':'test-model'})
            with self.assertRaises(GatewayError) as caught:
                gateway.complete([],'instructions',[],threading.Event(),lambda *_:None)
            self.assertEqual(caught.exception.code,'ConnectionFailed')
            self.assertEqual(caught.exception.public()['transport_error'],'ConnectError')
            self.assertIn('DNS',str(caught.exception))
            self.assertNotIn('private',json.dumps(caught.exception.public()))

    def test_quota_authentication_and_cancellation_do_not_retry(self):
        import httpx
        for status in (401,429):
            attempts=[]
            def request(req):
                attempts.append(req)
                return httpx.Response(status,json={'error':{}},headers={'x-request-id':'req-denied'})
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}):
                gateway=ModelGateway(Path(directory)/'settings.json',client_factory=lambda:httpx.Client(transport=httpx.MockTransport(request)))
                gateway.configure({'provider':'api','model':'test-model'})
                with self.assertRaises(GatewayError):
                    gateway.complete([],'instructions',[],threading.Event(),lambda *_:None)
                self.assertEqual(len(attempts),1)
        cancel=threading.Event()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'OPENAI_API_KEY':'paid-secret'}):
            def request(req):
                raise httpx.ReadTimeout('slow')
            gateway=ModelGateway(Path(directory)/'settings.json',client_factory=lambda:httpx.Client(transport=httpx.MockTransport(request)))
            gateway.configure({'provider':'api','model':'test-model'})
            def on_event(kind,data):
                if kind=='request_retry':cancel.set()
            with self.assertRaises(GatewayError) as caught:
                gateway.complete([],'instructions',[],cancel,on_event)
            self.assertEqual(caught.exception.code,'Canceled')


class MathTests(unittest.TestCase):
    def test_logic_finds_counterexample_and_handles_implication(self):
        disjunction={'op':'or','args':[{'op':'not','args':['P']},'Q']}
        self.assertTrue(truth_table({'op':'implies','args':['P','Q']},disjunction)['equivalent'])
        counterexample=truth_table({'op':'and','args':['P','Q']},disjunction)['counterexample']
        self.assertEqual(counterexample,{'P':False,'Q':False})
        with self.assertRaises(ValueError):truth_table({'op':'exec','args':['os','system']})

    def test_nested_sets_are_not_flattened(self):
        self.assertTrue(finite_sets('member',1,[1])['result'])
        self.assertFalse(finite_sets('member',[1],[1])['result'])
        self.assertTrue(finite_sets('member',[1],[[1]])['result'])
        self.assertEqual(finite_sets('intersection',[1,2],[2,3])['result'],[2])
        with self.assertRaises(ValueError):finite_sets('union',[True],[])


if __name__=='__main__':
    unittest.main()

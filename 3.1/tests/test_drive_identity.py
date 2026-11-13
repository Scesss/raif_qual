import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import app


class DriveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = app.Store(self.root / 'db.sqlite3')
        self.person = self.store.create_student({'email':'student@example.test','full_name':'Учебный Студент'})
        self.store.bind_drive_account('10000000000000000001', self.person['email'], 'Confirmed by teacher')

    def tearDown(self):
        self.tmp.cleanup()

    def test_binding_survives_new_store_and_homework_without_email(self):
        store = app.Store(self.root / 'db.sqlite3')
        for number in (2,3):
            hw = store.create_homework({'number':number})
            folder = self.root / str(number);folder.mkdir()
            (folder/'без имени.pdf').write_bytes(b'%PDF-1.7 test')
            metadata = {'без имени.pdf':{'file_id':str(number),'last_editor_permission_id':'10000000000000000001'}}
            with patch.object(app,'DATA',self.root/'data'):
                store.import_files(hw['id'],str(folder),metadata)
            row = store.list_reviews(hw['id'])[0]
            self.assertIsNotNone(row['submission_id'])
            self.assertEqual(row['match_status'],'drive_matched')

    def test_conflicts_and_display_name_are_not_silently_accepted(self):
        other = self.store.create_student({'email':'other@test.test','full_name':'Петров Пётр Петрович'})
        with self.assertRaises(ValueError):
            self.store.bind_drive_account('10000000000000000001',other['email'],'Wrong reassignment')
        person,status,_ = self.store.resolve_drive_identity({'last_editor_permission_id':'10000000000000000001'}, 'Петров Пётр Петрович.pdf')
        self.assertIsNone(person);self.assertEqual(status,'ambiguous')
        person,status,_ = self.store.resolve_drive_identity({'last_editor_name':'student'}, 'дм3.pdf')
        self.assertIsNone(person)
        person,status,_ = self.store.resolve_drive_identity({'owner_email':'lyudasha.2006@gmail.com','last_editor_permission_id':'10000000000000000001'},'дм3.pdf')
        self.assertEqual(person['id'],self.person['id'])
        person,status,_ = self.store.resolve_drive_identity({'last_editor_email':other['email'],'last_editor_permission_id':'10000000000000000001'},'дм3.pdf')
        self.assertEqual(status,'ambiguous')

    def test_new_preferred_format_cannot_replace_reviewed_file(self):
        hw = self.store.create_homework({'number':2})
        folder = self.root/'in';folder.mkdir()
        name='Учебный Студент'
        (folder/(name+'.pdf')).write_bytes(b'%PDF-1.7 test')
        with patch.object(app,'DATA',self.root/'data'):
            self.store.import_files(hw['id'],str(folder))
            row=self.store.list_reviews(hw['id'])[0]
            self.store.save_draft(row['id'],{'scores':[{'problem':1,'score':4,'reason':'Correct'}],'student_comment':'Верно.',
                'evidence':[{'source':'work','page':1,'note':'Решение проверено.'}]})
            for status in ('draft','approved'):
                if status=='approved':self.store.approve(row['id'],{})
                (folder/(name+status+'.tex')).write_text('New alternative')
                self.store.import_files(hw['id'],str(folder))
                current=self.store.review(row['id'])
                self.assertEqual(current['submission_id'],row['submission_id'])
                self.assertEqual(current['status'],status)

if __name__=='__main__':unittest.main()

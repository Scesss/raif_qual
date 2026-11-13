/**
 * Export a Google Drive folder tree to a Google Sheet with file metadata.
 *
 * Use case: homework files whose names do not contain student names.
 * The sheet shows both the owner and the last editor so they cannot be confused.
 *
 * Setup:
 *   1. In script.google.com create a standalone Apps Script project.
 *   2. Paste this whole file into Code.gs.
 *   3. In Services (+), enable "Drive API", version v2.
 *   4. Change ROOT_FOLDER_ID and STAFF_EMAILS below.
 *   5. Run exportDriveMetadata(); approve permissions once.
 */

const ROOT_FOLDER_ID = 'PASTE_DRIVE_FOLDER_ID_HERE';

// Accounts of teachers/assistants. They are never selected as a student email.
// Add all staff accounts that can appear as the last editor or shared-with user.
const STAFF_EMAILS = [
  'lyudasha.2006@gmail.com',
  'kiriell55@gmail.com',
];

const OUTPUT_SHEET_NAME = 'Drive metadata export';

function exportDriveMetadata() {
  if (ROOT_FOLDER_ID === 'PASTE_DRIVE_FOLDER_ID_HERE') {
    throw new Error('Set ROOT_FOLDER_ID before running the script.');
  }

  const rows = [[
    'relative_path', 'file_name', 'file_id', 'mime_type', 'size_bytes',
    'modified_at', 'owner_name', 'owner_email', 'last_editor_name',
    'last_editor_email', 'owner_is_staff', 'suggested_student_email',
    'email_source', 'web_url', 'owner_permission_id', 'last_editor_permission_id'
  ]];

  walkFolder_(ROOT_FOLDER_ID, '', rows);

  // A standalone project has no active spreadsheet. Create one on first run;
  // a bound script writes to its host spreadsheet instead.
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet() ||
    SpreadsheetApp.create('Drive metadata export');
  let sheet = spreadsheet.getSheetByName(OUTPUT_SHEET_NAME);
  if (!sheet) sheet = spreadsheet.insertSheet(OUTPUT_SHEET_NAME);
  if (sheet.getFilter()) sheet.getFilter().remove();
  sheet.clearContents();
  // Long account IDs must remain strings, including leading zeros.
  sheet.getRange(1, 15, rows.length, 2).setNumberFormat('@');
  sheet.getRange(1, 1, rows.length, rows[0].length).setValues(rows);
  sheet.setFrozenRows(1);
  sheet.getRange(1, 1, 1, rows[0].length).setFontWeight('bold');
  sheet.autoResizeColumns(1, rows[0].length);
  sheet.getRange(1, 1, rows.length, rows[0].length).createFilter();

  Logger.log('Exported %s files to %s', rows.length - 1, spreadsheet.getUrl());
}

function walkFolder_(folderId, parentPath, rows) {
  let pageToken = null;
  do {
    const result = Drive.Files.list({
      q: "'" + folderId + "' in parents and trashed = false",
      maxResults: 1000,
      pageToken: pageToken,
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
      fields: 'nextPageToken,items(id,title,mimeType,fileSize,modifiedDate,owners(displayName,emailAddress,permissionId),lastModifyingUser(displayName,emailAddress,permissionId),alternateLink)'
    });

    (result.items || []).forEach(function(file) {
      const path = parentPath ? parentPath + '/' + file.title : file.title;
      if (file.mimeType === 'application/vnd.google-apps.folder') {
        walkFolder_(file.id, path, rows);
        return;
      }

      const owner = (file.owners || [])[0] || {};
      const editor = file.lastModifyingUser || {};
      const ownerEmail = normalizeEmail_(owner.emailAddress);
      const editorEmail = normalizeEmail_(editor.emailAddress);
      const ownerIsStaff = isStaff_(ownerEmail);

      // Correct priority: non-staff owner -> non-staff last editor -> empty.
      // Do not use a staff email as the student identity.
      let candidate = '';
      let source = '';
      if (ownerEmail && !ownerIsStaff) {
        candidate = ownerEmail;
        source = 'owner';
      } else if (editorEmail && !isStaff_(editorEmail)) {
        candidate = editorEmail;
        source = 'last_editor_fallback';
      }

      rows.push([
        parentPath,
        file.title || '',
        file.id || '',
        file.mimeType || '',
        file.fileSize || '',
        file.modifiedDate || '',
        owner.displayName || '',
        ownerEmail,
        editor.displayName || '',
        editorEmail,
        ownerIsStaff ? 'yes' : 'no',
        candidate,
        source,
        file.alternateLink || '',
        String(owner.permissionId || ''),
        String(editor.permissionId || '')
      ]);
    });
    pageToken = result.nextPageToken;
  } while (pageToken);
}

function normalizeEmail_(value) {
  return String(value || '').trim().toLowerCase();
}

function isStaff_(email) {
  return STAFF_EMAILS.indexOf(normalizeEmail_(email)) !== -1;
}

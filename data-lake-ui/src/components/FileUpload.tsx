import React, { useRef, useState } from 'react';
import { uploadFile } from '../api';

type Props = {
  tableName: string;
  onUploaded: () => void;
};

const FileUpload: React.FC<Props> = ({ tableName, onUploaded }) => {
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith('.xlsx')) {
      setError('Only .xlsx files are supported');
      return;
    }
    setUploading(true);
    setError(null);
    setSuccess(false);
    try {
      await uploadFile(tableName, file);
      setSuccess(true);
      onUploaded();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  };

  return (
    <div style={{ marginTop: 24 }}>
      <label style={{ fontWeight: 500 }}>Upload .xlsx file:</label>
      <input
        type="file"
        accept=".xlsx"
        ref={fileInput}
        disabled={uploading}
        onChange={handleFileChange}
        style={{ marginLeft: 12 }}
      />
      {uploading && <span style={{ marginLeft: 12 }}>Uploading...</span>}
      {success && <span style={{ color: 'green', marginLeft: 12 }}>Uploaded!</span>}
      {error && <span style={{ color: 'red', marginLeft: 12 }}>{error}</span>}
    </div>
  );
};

export default FileUpload; 
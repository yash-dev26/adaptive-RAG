import { useState, useCallback } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

export function useFileUpload({ buildHeaders, addEntry, hasOpenAiKey, openModal }) {
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [hasUploadedFile, setHasUploadedFile] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [ingestedFileId, setIngestedFileId] = useState(null);

  const uploadFile = useCallback(
    async (file) => {
      if (!file) return;

      if (!hasOpenAiKey) {
        openModal();
        return;
      }

      setUploadedFileName(file.name);
      setIsUploading(true);

      try {
        const formData = new FormData();
        formData.append("file", file);

        const { data } = await axios.post(`${API_BASE}/upload/`, formData, {
          headers: buildHeaders(),
        });

        setHasUploadedFile(true);
        setIngestedFileId(data.file_id);

        addEntry({
          query: `Uploaded: ${file.name}`,
          answer: `File ingested. ID: \`${data.file_id}\``,
          sources: [],
          confidence: null,
          pipeline: [{ name: "ingest", status: "done", detail: `Ingested "${file.name}"`, badge: null }],
          cacheHit: false,
        });
      } catch (err) {
        const message = err?.response?.data?.detail || err.message || "Upload failed";
        setHasUploadedFile(false);
        setIngestedFileId(null);
        if (err?.response?.status === 401) openModal();
        addEntry({
          query: `Uploaded: ${file.name}`,
          answer: `Upload failed: ${message}`,
          sources: [],
          confidence: 0,
          pipeline: [{ name: "ingest", status: "error", detail: message, badge: null }],
          cacheHit: false,
        });
      } finally {
        setIsUploading(false);
      }
    },
    [buildHeaders, addEntry, hasOpenAiKey, openModal]
  );

  const resetUpload = useCallback(() => {
    setUploadedFileName(null);
    setHasUploadedFile(false);
    setIngestedFileId(null);
  }, []);

  // Used when switching into an existing thread that already has a file
  // attached — no upload happens, just restoring the display state.
  const applyThreadFile = useCallback((thread) => {
    setIngestedFileId(thread.file_id || null);
    setHasUploadedFile(Boolean(thread.file_id));
    setUploadedFileName(
      thread.file_name || (thread.file_id ? `Document ${thread.file_id.slice(0, 8)}` : null)
    );
  }, []);

  return {
    uploadedFileName,
    hasUploadedFile,
    isUploading,
    ingestedFileId,
    uploadFile,
    resetUpload,
    applyThreadFile,
  };
}
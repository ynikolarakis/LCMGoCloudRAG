import { UploadForm } from "@/components/documents/UploadForm";
import { DocumentList } from "@/components/documents/DocumentList";

export default function DocumentsPage() {
  return (
    <div className="p-6 space-y-6" data-testid="documents-page">
      <UploadForm />
      <DocumentList />
    </div>
  );
}

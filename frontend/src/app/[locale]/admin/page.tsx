import { AdminPage } from "@/components/admin/AdminPage";
import { AuthGuard } from "@/components/AuthGuard";

export default function AdminPageRoute() {
  return (
    <AuthGuard requiredRole="admin">
      <AdminPage />
    </AuthGuard>
  );
}

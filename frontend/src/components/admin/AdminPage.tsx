"use client";

import { useTranslations } from "next-intl";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { HealthPanel } from "./HealthPanel";
import { AuditLogTable } from "./AuditLogTable";

export function AdminPage() {
  const t = useTranslations("admin");

  return (
    <div className="p-6" data-testid="admin-page">
      <h2 className="text-2xl font-semibold mb-6">{t("title")}</h2>

      <Tabs defaultValue="health" data-testid="admin-tabs">
        <TabsList>
          <TabsTrigger value="health" data-testid="tab-health">
            {t("healthTab")}
          </TabsTrigger>
          <TabsTrigger value="audit" data-testid="tab-audit">
            {t("auditTab")}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="health" className="mt-4">
          <HealthPanel />
        </TabsContent>

        <TabsContent value="audit" className="mt-4">
          <AuditLogTable />
        </TabsContent>
      </Tabs>
    </div>
  );
}

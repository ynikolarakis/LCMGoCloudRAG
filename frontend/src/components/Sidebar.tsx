"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { FileText, MessageSquare, Settings, Menu, X, Globe } from "lucide-react";
import { Link, usePathname } from "@/i18n/routing";
import { useLocale } from "next-intl";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

const navItems = [
  { href: "/chat" as const, icon: MessageSquare, labelKey: "nav.chat" as const },
  { href: "/documents" as const, icon: FileText, labelKey: "nav.documents" as const },
  { href: "/admin" as const, icon: Settings, labelKey: "nav.admin" as const },
];

export function Sidebar() {
  const t = useTranslations();
  const pathname = usePathname();
  const locale = useLocale();
  const [mobileOpen, setMobileOpen] = useState(false);
  const otherLocale = locale === "en" ? "el" : "en";

  return (
    <>
      {/* Mobile hamburger */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
        data-testid="sidebar-toggle"
      >
        {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r bg-background transition-transform md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
        data-testid="sidebar"
      >
        <div className="flex h-16 items-center px-6">
          <h1 className="text-lg font-semibold" data-testid="app-title">
            {t("common.appName")}
          </h1>
        </div>

        <Separator />

        <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
          {navItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )}
                onClick={() => setMobileOpen(false)}
                data-testid={`nav-${item.href.slice(1)}`}
              >
                <item.icon className="h-4 w-4" />
                {t(item.labelKey)}
              </Link>
            );
          })}
        </nav>

        <Separator />

        {/* Locale switcher */}
        <div className="p-3">
          <Link
            href={pathname || "/chat"}
            locale={otherLocale}
            className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            data-testid="locale-switcher"
          >
            <Globe className="h-4 w-4" />
            {otherLocale === "el" ? "Ελληνικά" : "English"}
          </Link>
        </div>
      </aside>
    </>
  );
}

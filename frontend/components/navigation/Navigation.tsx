"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button, Tab, TabList } from "@tremor/react";
import { HomeIcon, BarChart3Icon, CircleDollarSignIcon, UserIcon, LogInIcon, LogOutIcon } from "lucide-react"; // Assuming lucide-react for icons
// import { useAuth } from "@/stores/authStore"; // Placeholder for auth

export default function Navigation() {
  const pathname = usePathname();
  // const { user, logout } = useAuth(); // Placeholder for auth
  const user = null; // Placeholder for auth
  const logout = () => console.log("logout"); // Placeholder for auth

  const navItems = [
    { name: "Dashboard", href: "/", icon: HomeIcon },
    { name: "Horse Racing", href: "/horse-racing", icon: BarChart3Icon },
    { name: "Football", href: "/football", icon: CircleDollarSignIcon },
    // Add more items as needed
  ];

  return (
    <nav className="bg-white shadow-md dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="text-2xl font-bold text-tremor-brand dark:text-dark-tremor-brand">
            BettingSystem
          </Link>
          <div className="flex items-center space-x-4">
            <TabList variant="solid" className="hidden sm:flex">
              {navItems.map((item) => (
                <Link href={item.href} key={item.name} passHref legacyBehavior>
                  <Tab
                    icon={item.icon}
                    className={
                      pathname === item.href
                        ? "text-tremor-brand-strong dark:text-dark-tremor-brand-strong"
                        : "text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
                    }
                  >
                    {item.name}
                  </Tab>
                </Link>
              ))}
            </TabList>
            <div className="flex items-center space-x-2">
              {user ? (
                <>
                  <Button icon={UserIcon} variant="secondary" className="hidden sm:flex">
                    {/* @ts-ignore */}
                    {user.username}
                  </Button>
                  <Button icon={LogOutIcon} onClick={logout} variant="light">
                    Logout
                  </Button>
                </>
              ) : (
                <Link href="/login" passHref legacyBehavior>
                  <Button icon={LogInIcon} variant="primary">
                    Login
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
      {/* Mobile navigation tabs */}
      <div className="sm:hidden border-t">
         <TabList variant="solid" className="w-full grid grid-cols-3">
            {navItems.map((item) => (
            <Link href={item.href} key={item.name + "-mobile"} passHref legacyBehavior>
                <Tab
                icon={item.icon}
                className={
                    pathname === item.href
                    ? "text-tremor-brand-strong dark:text-dark-tremor-brand-strong"
                    : "text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
                }
                >
                {item.name}
                </Tab>
            </Link>
            ))}
        </TabList>
      </div>
    </nav>
  );
}

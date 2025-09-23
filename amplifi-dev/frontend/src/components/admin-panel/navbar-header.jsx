"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Bell, Settings } from "lucide-react";
import Image from "next/image";
import Logo from "../../assets/icons/Amplifi2.svg";
import noUser from "@/assets/icons/noUser.png";
import { useUser } from "@/context_api/userContext";
import { useState, useEffect } from "react";
import { logout } from "@/api/login";
import { constants } from "@/lib/constants";
import { getCookie, removeCookie } from "@/utils/cookieHelper";
import NewAvatar from "../ui/newAvatar";

const Navbar = () => {
  const pathname = usePathname();
  const { user, setUser } = useUser();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [activeNav, setActiveNav] = useState("");
  const [navbarLinksVisible, setNavbarLinksVisible] = useState(true);
  const router = useRouter();
  useEffect(() => {
    console.log(user);
  }, [user]);
  // Function to handle logout
  const handleLogout = async () => {
    try {
      let response = null;
      if (getCookie(constants.AUTH_TOKEN)) {
        response = await logout();
      }
      if (response == null || response.status === 204) {
        removeCookie(constants.AUTH_TOKEN);
        removeCookie(constants.REFRESH_TOKEN);
        removeCookie(constants.JWT_TOKEN);
        localStorage.removeItem(constants.USER);
        setUser(null);
        router.push("/login"); // Redirect to login
      }
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const navItems = [
    { name: "Workspace", href: `/workspace/?id=${user?.clientId}` },
    { name: "Destination", href: `/destination/?id=${user?.clientId}` },
    { name: "Workflows", href: `/workflows/?id=${user?.clientId}` },
  ];

  // Check and set the active navigation link on render and when the pathname changes
  useEffect(() => {
    if (navItems && pathname) {
      const normalizedPathname = pathname.replace(/\/$/, "");

      const currentActive = navItems.find((item) => {
        const baseHref = item.href.split("?")[0].replace(/\/$/, "");
        return normalizedPathname === baseHref;
      });

      if (currentActive) {
        setActiveNav(currentActive.href);
      }
    }
  }, [pathname, navItems]);

  useEffect(() => {
    const isChatHostname = process.env.NEXT_PUBLIC_CHAT_HOST_NAME.includes(
      window.location.hostname,
    );
    const isChatPage = pathname.includes("/chatapp");

    setNavbarLinksVisible(!isChatHostname && !isChatPage);
  }, [pathname]);

  const isActive = (href) => {
    return activeNav === href
      ? "text-blue-700 bg-custom-headerColor"
      : "text-gray-500 hover:text-gray-700";
  };

  return (
    <div>
      <nav className="bg-white dark:bg-gray-900 fixed w-full z-20 top-0 start-0 border-b border-gray-200 dark:border-gray-600">
        <div className="max-w-full px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-[54px]">
            <div className="flex items-center">
              <Link
                href={`/workspace/?id=${user?.clientId}`}
                className="flex-shrink-0"
              >
                <div className="flex items-center h-full">
                  <Image
                    src={Logo}
                    alt="logo"
                    width={80}
                    height={26}
                    className="h-full w-auto object-contain"
                  />
                </div>
              </Link>
              {navbarLinksVisible && (
                <div className="hidden md:block ml-[92px]">
                  <div className="flex items-baseline space-x-4">
                    {navItems.map((item) => (
                      <Link
                        key={item.name}
                        href={item.href}
                        className={`px-3 py-2 rounded-md text-sm font-medium ${isActive(
                          item.href,
                        )}`}
                      >
                        {item.name}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="hidden md:block">
              <div className="ml-4 flex items-center md:ml-6">
                {navbarLinksVisible && (
                  <>
                    {/* <Button variant="ghost" size="icon">
                      <Bell className="h-5 w-5" />
                    </Button> */}
                    <Link href={"/users"}>
                      <Button variant="ghost" size="icon">
                        <Settings
                          className={`${
                            pathname === "/users"
                              ? "text-blue-600"
                              : "text-gray-700"
                          } h-5 w-5`}
                        />
                      </Button>
                    </Link>
                  </>
                )}
                <div className="ml-3 flex items-center">
                  <span className="text-sm font-semibold text-gray-700 mr-4">
                    {user?.fullName}
                  </span>
                  <NewAvatar
                    title={user?.fullName}
                    onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    className="cursor-pointer"
                  />
                  {/* Dropdown */}
                  {isDropdownOpen && (
                    <div className="absolute right-0 top-14 w-48 bg-white border border-gray-200 shadow-lg rounded-md z-10 transition-all duration-300 ease-in-out">
                      <div
                        className="px-4 py-2 text-sm text-gray-700 cursor-pointer"
                        onClick={handleLogout}
                      >
                        Logout
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="md:hidden">
              <Button variant="ghost" size="icon">
                <svg
                  className="h-6 w-6"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </Button>
            </div>
          </div>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;

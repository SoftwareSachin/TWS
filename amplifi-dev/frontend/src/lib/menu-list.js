import { Users, Settings, LayoutGrid, UserCog } from "lucide-react";
import { ROLES } from "./constants";

export function getMenuList(pathname) {
  return [
    {
      groupLabel: "",
      menus: [
        {
          href: "/users",
          label: "Users",
          icon: Users,
        },
        // {
        //   href: "/billing",
        //   label: "Billing",
        //   icon: Settings
        // },
        {
          href: "/deployment",
          label: "Deployment",
          icon: Users,
        },
        {
          href: "/documentation",
          label: "Documentation",
          icon: Settings,
        },
        {
          href: "/api-clients",
          label: "Manage API Clients",
          icon: UserCog,
          roles: [ROLES.ADMIN], // Only for Amplifi Admin
        },
      ],
    },
  ];
}

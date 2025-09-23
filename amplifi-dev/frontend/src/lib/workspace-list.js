import McpIcon from "@/assets/icons/MCP.svg";
import ToolsIcon from "@/assets/icons/tools.svg";
import Agents from "@/assets/icons/agents.svg";
import FileText from "@/assets/icons/file2.svg";
import Box from "@/assets/icons/dataset.svg";
import Search from "@/assets/icons/search-md.svg";
import MessageSquareText from "@/assets/icons/chat.svg";
import UsersRound from "@/assets/icons/users.svg";

export function workspaceMenuList(id, pathname) {
  return [
    {
      groupLabel: "",
      menus: [
        {
          href: `/workspace/${id}/files/0`,
          label: "Files",
          icon: FileText,
        },
        {
          href: `/workspace/${id}/datasets`,
          label: "Datasets",
          icon: Box,
        },
        {
          href: `/workspace/${id}/search`,
          label: "Search",
          icon: Search,
        },
        {
          href: `/workspace/${id}/mcp`,
          label: "MCP",
          icon: McpIcon,
        },
        {
          href: `/workspace/${id}/tools`,
          label: "Tools",
          icon: ToolsIcon,
        },
        {
          href: `/workspace/${id}/agents`,
          label: "Agents",
          icon: Agents,
        },
        {
          href: `/workspace/${id}/chat`,
          label: "Chats",
          icon: MessageSquareText,
        },
        {
          href: `/workspace/${id}/users`,
          label: "Users",
          icon: UsersRound,
        },
      ],
    },
  ];
}

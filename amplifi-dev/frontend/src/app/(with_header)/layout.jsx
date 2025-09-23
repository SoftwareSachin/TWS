/**
 * The layout function renders a page layout with a Navbar component and children components.
 * @returns The layout component is being returned, which includes a div with a gray background color
 * and a height of the screen. Inside the div, the Navbar component is rendered followed by a div with
 * top margin, padding, and children components passed as props.
 */
import Navbar from "../../components/admin-panel/navbar-header";

const layout = ({ children }) => {
  return (
    <div className="flex flex-col h-screen bg-[#F4F5F7]">
      <Navbar />
      <div className="mt-[54px] flex-1 flex flex-col overflow-hidden">
        <div className="overflow-y-auto px-0">{children}</div>
      </div>
    </div>
  );
};

export default layout;

import * as React from "react"
import { cn } from "@/lib/utils"
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu"
import { Home, Zap, Github, Info } from "lucide-react" 

const dropdownComponents: { title: string; href: string; description: string; icon: React.ReactNode }[] = [
  {
    title: "Housing",
    href: "/housing",
    description: "Analysis of prices and trends in the real estate market.",
    icon: <Home className="h-4 w-4 text-sky-500" />,
  },
  {
    title: "Electricity",
    href: "/electricity",
    description: "Modeling and prediction of electricity consumption.",
    icon: <Zap className="h-4 w-4 text-yellow-500" />,
  },
]

interface ListItemProps extends React.ComponentPropsWithoutRef<"a"> {
    icon?: React.ReactNode;
}

const ListItem = React.forwardRef<
  React.ElementRef<"a">,
  ListItemProps
>(({ className, title, children, icon, ...props }, ref) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          ref={ref}
          className={cn(
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors",
            "hover:bg-slate-100 focus:bg-slate-100 dark:hover:bg-slate-800 dark:focus:bg-slate-800",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none flex items-center space-x-2 text-slate-900 dark:text-slate-50">
            {icon}
            <span>{title}</span>
          </div>
          <p className="line-clamp-2 text-sm leading-snug text-slate-500 dark:text-slate-400">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  )
})
ListItem.displayName = "ListItem"


const NavBar = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white shadow-md dark:border-slate-800 dark:bg-slate-900">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        
        <div className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400 flex items-center space-x-1">

          <span>Najebali smo</span>
        </div>

        <NavigationMenu className="flex-grow flex justify-end">
          <NavigationMenuList className="flex space-x-2">

            <NavigationMenuItem>
              <NavigationMenuTrigger 
                className="flex items-center space-x-1 font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <Zap className="h-4 w-4 mr-1 text-indigo-500" />
                <span>Models</span>
              </NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] lg:w-[600px] lg:grid-cols-2">
                  {dropdownComponents.map((component) => (
                    <ListItem
                      key={component.title}
                      title={component.title}
                      href={component.href}
                      icon={component.icon}
                    >
                      {component.description}
                    </ListItem>
                  ))}
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <NavigationMenuLink 
                href="https://github.com/Jfgm299/Machine_Learning_AWS" 
                target="_blank" 
                className={cn(navigationMenuTriggerStyle(), "font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 px-4")} 
              >
                <div className="flex items-center">
                    <Github className="h-4 w-4 mr-2" />
                    GitHub
                </div>
              </NavigationMenuLink>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <NavigationMenuLink 
                href="/info" 
                className={cn(navigationMenuTriggerStyle(), "font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 px-4")}
              >
                <div className="flex items-center">
                    <Info className="h-4 w-4 mr-2" />
                    Info
                </div>
              </NavigationMenuLink>
            </NavigationMenuItem>

          </NavigationMenuList>
        </NavigationMenu>
      </div>
    </header>
  )
}

export default NavBar;
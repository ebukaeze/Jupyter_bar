"use client";
import { navLinks } from "@/constants";
import Image from "next/image";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";

const Navbar = () => {
  useGSAP(() => {
    const navTween = gsap.timeline({
      scrollTrigger: {
        trigger: "nav",
        start: "bottom top",
      },
    });

    navTween.fromTo(
      "nav",
      { backgroundColor: "transparent" },
      {
        backgroundColor: "#00000050",
        backgroundFilter: "blur(10px)",
        duration: 1,
        ease: "power1.inOut",
      },
    );
  });

  return (
    <div className="flex items-center gap-2 justify-between p-4 nav">
      <div className="flex gap-2">
        <Image src={"/images/logo.png"} alt="Jupyter" height={16} width={16} />
        <p className="text-lg font-bold">Jupyter Bar</p>
      </div>
      <ul className="flex gap-x-4">
        {navLinks.map((item) => (
          <li key={item.id} className="">
            <a href={`#${item.id}`}>{item.title}</a>{" "}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Navbar;

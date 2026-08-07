import Image from "next/image";
import React from "react";

const Hero = () => {
  return (
    <section id="hero" className="noisy">
      <h1 className="title">Jupyter</h1>

      <Image
        src="/images/hero-left-leaf.png"
        alt="left leaf"
        className="left-leaf"
        width={120}
        height={120}
      />
      <Image
        src="/images/hero-right-leaf.png"
        alt="right leaf"
        className="right-leaf"
        width={112}
        height={112}
      />
      <div className="body">
        <div className="content">
          <div className="space-y-5 hidden md:block">
            <p>Cool. Crisp. Classic</p>
            <p>
              Sip the Spirit <br /> of Summer
            </p>
          </div>
          <div className="view-cocktails">
            <p className="subtitle">
              Every cocktail on our menu is a blend of premium ingredients,
              creative flair, and timeless recipes - designed to delight your
              senses
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

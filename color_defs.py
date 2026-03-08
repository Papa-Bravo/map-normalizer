# coding: utf-8

from PIL import Image

def convert_html(hex_str,normed=False):
    r=eval("0x"+hex_str[0:2])
    g=eval("0x"+hex_str[2:4])
    b=eval("0x"+hex_str[4:6])
    if normed:
        r/=255.
        g/=255.
        b/=255.
    return(r,g,b)

white=convert_html("ffffff")
black=convert_html("000000")
deep_sea=convert_html("e1f1fa")
shallow_sea=convert_html("c5e5f5")
yel_light=convert_html("f0cb4b")
green_light=convert_html("3fa73b")
red_light=convert_html("e4363e")
yel_bank=convert_html("fffad8")
yel_build=convert_html("e5dfc4")
blue_line=convert_html("71b2d0")
purple_txt=convert_html("d223ba")
pink_vts=convert_html("fdcee0")
green_dry=convert_html("bbc1a7")

scan_white=convert_html("fdfdfd")
scan_black=convert_html("4e4b43")
scan_deep_sea=convert_html("f4fbfb")
scan_shallow_sea=convert_html("e2f3fb")
scan_green_light=convert_html("83ac77")
scan_red_light=convert_html("e37478")
scan_yel_light=convert_html("cca46c")
scan_yel_bank=convert_html("fdfbc7")
scan_yel_build=convert_html("ddd2a6")
scan_blue_line=convert_html("a1cdf8")
scan_purple_txt=convert_html("b97496")
scan_pink_vts=convert_html("fddbe9")
scan_green_dry=convert_html("bbc1a7")


colors=[white,
        black,
        deep_sea,
        shallow_sea,
        yel_light,
        green_light,
        red_light,
        yel_bank,
        yel_build,
        blue_line,
        purple_txt,
        pink_vts,
        green_dry]

scan_colors=[scan_white,
             scan_black,
             scan_deep_sea,
             scan_shallow_sea,
             scan_yel_light,
             scan_green_light,
             scan_red_light,
             scan_yel_bank,
             scan_yel_build,
             scan_blue_line,
             scan_purple_txt,
             scan_pink_vts,
             scan_green_dry]

if __name__=="__main__":
    from matplotlib import pyplot as plt

    def norm_colors(cols,denom=255.):
        for (r,g,b) in cols:
            yield r/denom,g/denom,b/denom
        
    ax = plt.figure().add_subplot(projection='3d')

    rs=[c[0] for c in colors]
    gs=[c[1] for c in colors]
    bs=[c[2] for c in colors]
    ax.scatter(rs,gs,bs,c=list(norm_colors(colors)))

    rss=[c[0] for c in scan_colors]
    gss=[c[1] for c in scan_colors]
    bss=[c[2] for c in scan_colors]
    ax.scatter(rss,gss,bss,c=list(norm_colors(scan_colors)))

    ax.set_xlabel("red")
    ax.set_ylabel("green")
    ax.set_zlabel("blue")
    for (r1,g1,b1,r2,g2,b2) in zip(rs,gs,bs,rss,gss,bss):
        plt.plot([r1,r2],[g1,g2],[b1,b2],"k")
    plt.show()
    

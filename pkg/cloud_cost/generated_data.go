// Code generated by "esc -o generated_data.go -pkg cloud_cost -prefix _fixtures -private _fixtures"; DO NOT EDIT.

package cloud_cost

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"sync"
	"time"
)

type _escLocalFS struct{}

var _escLocal _escLocalFS

type _escStaticFS struct{}

var _escStatic _escStaticFS

type _escDirectory struct {
	fs   http.FileSystem
	name string
}

type _escFile struct {
	compressed string
	size       int64
	modtime    int64
	local      string
	isDir      bool

	once sync.Once
	data []byte
	name string
}

func (_escLocalFS) Open(name string) (http.File, error) {
	f, present := _escData[path.Clean(name)]
	if !present {
		return nil, os.ErrNotExist
	}
	return os.Open(f.local)
}

func (_escStaticFS) prepare(name string) (*_escFile, error) {
	f, present := _escData[path.Clean(name)]
	if !present {
		return nil, os.ErrNotExist
	}
	var err error
	f.once.Do(func() {
		f.name = path.Base(name)
		if f.size == 0 {
			return
		}
		var gr *gzip.Reader
		b64 := base64.NewDecoder(base64.StdEncoding, bytes.NewBufferString(f.compressed))
		gr, err = gzip.NewReader(b64)
		if err != nil {
			return
		}
		f.data, err = ioutil.ReadAll(gr)
	})
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (fs _escStaticFS) Open(name string) (http.File, error) {
	f, err := fs.prepare(name)
	if err != nil {
		return nil, err
	}
	return f.File()
}

func (dir _escDirectory) Open(name string) (http.File, error) {
	return dir.fs.Open(dir.name + name)
}

func (f *_escFile) File() (http.File, error) {
	type httpFile struct {
		*bytes.Reader
		*_escFile
	}
	return &httpFile{
		Reader:   bytes.NewReader(f.data),
		_escFile: f,
	}, nil
}

func (f *_escFile) Close() error {
	return nil
}

func (f *_escFile) Readdir(count int) ([]os.FileInfo, error) {
	if !f.isDir {
		return nil, fmt.Errorf(" escFile.Readdir: '%s' is not directory", f.name)
	}

	fis, ok := _escDirs[f.local]
	if !ok {
		return nil, fmt.Errorf(" escFile.Readdir: '%s' is directory, but we have no info about content of this dir, local=%s", f.name, f.local)
	}
	limit := count
	if count <= 0 || limit > len(fis) {
		limit = len(fis)
	}

	if len(fis) == 0 && count > 0 {
		return nil, io.EOF
	}

	return fis[0:limit], nil
}

func (f *_escFile) Stat() (os.FileInfo, error) {
	return f, nil
}

func (f *_escFile) Name() string {
	return f.name
}

func (f *_escFile) Size() int64 {
	return f.size
}

func (f *_escFile) Mode() os.FileMode {
	return 0
}

func (f *_escFile) ModTime() time.Time {
	return time.Unix(f.modtime, 0)
}

func (f *_escFile) IsDir() bool {
	return f.isDir
}

func (f *_escFile) Sys() interface{} {
	return f
}

// _escFS returns a http.Filesystem for the embedded assets. If useLocal is true,
// the filesystem's contents are instead used.
func _escFS(useLocal bool) http.FileSystem {
	if useLocal {
		return _escLocal
	}
	return _escStatic
}

// _escDir returns a http.Filesystem for the embedded assets on a given prefix dir.
// If useLocal is true, the filesystem's contents are instead used.
func _escDir(useLocal bool, name string) http.FileSystem {
	if useLocal {
		return _escDirectory{fs: _escLocal, name: name}
	}
	return _escDirectory{fs: _escStatic, name: name}
}

// _escFSByte returns the named file from the embedded assets. If useLocal is
// true, the filesystem's contents are instead used.
func _escFSByte(useLocal bool, name string) ([]byte, error) {
	if useLocal {
		f, err := _escLocal.Open(name)
		if err != nil {
			return nil, err
		}
		b, err := ioutil.ReadAll(f)
		_ = f.Close()
		return b, err
	}
	f, err := _escStatic.prepare(name)
	if err != nil {
		return nil, err
	}
	return f.data, nil
}

// _escFSMustByte is the same as _escFSByte, but panics if name is not present.
func _escFSMustByte(useLocal bool, name string) []byte {
	b, err := _escFSByte(useLocal, name)
	if err != nil {
		panic(err)
	}
	return b
}

// _escFSString is the string version of _escFSByte.
func _escFSString(useLocal bool, name string) (string, error) {
	b, err := _escFSByte(useLocal, name)
	return string(b), err
}

// _escFSMustString is the string version of _escFSMustByte.
func _escFSMustString(useLocal bool, name string) string {
	return string(_escFSMustByte(useLocal, name))
}

var _escData = map[string]*_escFile{

	"/aws_instances.json": {
		name:    "aws_instances.json",
		local:   "_fixtures/aws_instances.json",
		size:    140630,
		modtime: 1559452593,
		compressed: `
H4sIAAAAAAAC/+x9S28jydHtfn4FofU4kRH5np3hC881YAMNXNsb4y6KrGKP8LUekNjwPDD//YPUj5HI
alWcrGAp2c3NLEY6TUqKSkZGnMdv361WF7d3N/37ze7+4ofVf75brVar3x7/u1pdbLrd8Pbm7peLH1YX
Pw7Xw133bnX7/u725n64+P7TN+1+uR0evuEq9Ib453fd3dsnX725/j/DVXfdv7m73Dx8G5tE/Pmr97c3
u09f+c/H/7la/ba6+PXm+vEffX//p6G73/2JNhffP7zTD99KxnofVr9//yKkfwqxJjPTFGS7B6HIU5Du
OaQk9lOQ9R7E5bL6/SPi/3/+3Wxu39+/Ge7+fXXxw8rnz//7arj69H+p/PGbfPvku+3n/3u9+++b4W77
8OpkVz9evu3Wl7uL/S//5cmf+afLtz/98Q0Pb/2hLq7fv3v3+X92u93d5fr97vErv33+SR/e8MO/4PPF
Hz/+xeX1/a673gz//OV2+IugnD78hB+/iQo//cL1sPvvzd3/jL7lj9/1+x+/vvd3d8P17sfh+uKH1e7u
/fDdk6/XVngnqnAbfUWFr/Ha2+xXeHLoQyF4lfkPxQfIucLbr3Ay91fdu3dfrm77cPRWVPdeRVjr0WNS
AtngkH4M8mKp0milmiSv1L/f/PfFEn339Os1FUp6BWqSoEAf3vDL9bnt3t1PF+g/Hl93dXO7u7y6/HXo
Dyv0LnRm6gS2hjlW1Oh+KSQK4NloEznwbLQ5FvRJELzK/pNQYpg6gcfK2jV5/kqq+4ul9Ky83ZLHr6S6
DU9Xd7A1J/BeqVL0ESxVSjmApUopBrBUmVwED21KMU9U92h78aRPa6e6s151R99Sc7FjSXG7uvvhfqkS
of2FBLLBIT0OGXDIdgyCPxDIcf+Pm36463bDi4/D1dBfvr9a4IEQNTOi0/7jW55+Ij5/Zf3+7n4nfUz+
cnN1+343vPQpsAmCFoeSQheePKO9RywF7T0y3IQnimjnFUNVg5PlBf+v29XuZtVWl/PlanpW+LmxMYps
UugTz52jPP4jUCl9RJyHFa89rJAclF7UUZSs0C6nktB22SX0dOVEhPYgLqJDP2IH9yDsqaahoCB/JP7v
s1p/zWZCeKpS0GwnVB6IDZs8ebCOPg3PfolP/65PesKnVydrQoOn3fMOb+bf9+Fn1DnwZNOvH9/8a/Xp
LR/+bd86Q3Hqb+tNiHP7QjLecoSOoYeXhQ4hMu7hyg6cQWRCdFO3/Dja4/k83uX5sVJlWakOP+/uunm1
GkUd3mhRPCtTnyXt3Yc3vMg933Xmuru+eXGJYH2auwV++FcIn+/Dl2kJpMMh6zHIi9U9ehBD5/DRFw+s
1mzKzt7pxUP9TX16XLu5o+mPW/fkDzTjA5e9P9nPW9m0kr1f8tN2+s/7K/Vmeg6To8IgJoSMjitzROcw
ghfpcch2DIKfYxSbncQo1jjFpiYx3vjp+3OeX98uUALr27FH+S/OozdhF6ff195N2FFCt8KO2U1xCuLc
/dQit+fn5Tvr8120m9K7PEt2r725GnbdC+SXaErVcmrv6gN2oBJAhwLWzwFssan+h5d4saTLaEmnCEzc
OSx1FytR74xPcdH72P/b3dx1b18cC1266XOeDHuFk94Hy2jt+Qwe9J6mqbx7J72302PSvZPee4c2Pz5Z
X3XSE3Oz7Y/s0P9yGT5vgLgpwmNHklWrVWAkWOfght3BTJgPkG9uDSrqOHJjrbeAxjh/b2VjIHR4FQPM
CkgWPcNtDPDsLka0W7fR55rnAbmNLtJ56z0GoiuoXuM9sdQQccJimD8rpoIKhgSIAUZ0Iwi1tSq1vVad
XmXobVRlM0AZx2C6QdBoDyw6AZRANjikxyEDDtn/mHLENVNDZ1JokLKouYV++BGb4xkEU6YvlMHNF2GE
4sBSCoXRdV20CdVghGLRdj4UO3XYu/GZSYu8MhcVazy1JsPoDU9OTGqngXuzuhQ5gBOTROCwLhHIzkg+
RHBImeg8EGxiIChragRjj1IURHT4hkUCOdDd+QDfcgnW3UVH8C334WfB75/JhFO8gEqPe7NsRyMYkf9E
klVocXH+TdTBW0q0BXKoCNUdfz95ykPrtnR0V9OLymA8+/mtic0ZXeZIIBsc0uOQ7RikokFx2Z9ugyKT
xWW/ZH8i6L+dgMVuYs1ZvNdrCHgpe0dlELBMNgcvgk5dwqQG6cwSbICTf8uCProo8KdcQqeDznlUqOk4
erg1geXMznm0vS81LXSkxmbkSnT/SMvub6bbEdFdkhXo/hmWugm0xwdXyRBxC5cIL0yLP/Ze8mQX9YuS
ZCX0Kckpn6um3XuVxCGgQwkJZINDehwyjEEqLIcgRctiKx2veJ901iy86JySFt4LKjwFhQL3KcHV6hJa
rSjVj0vRrVVq3yBrutNwavpX2b3wcnN38/kt3Y8IC8lcPXzPi8JCnm+9aS3DRoMM6xAZliEyrELkGitD
ayI5+Qn87+Hul1UbjoZfqqA9YWEkp+dpWKctnLhWumlWlDNWQeltIzrfc9GD8z0XLEa9coK3tT/d81Tn
AAdcEhfup/OC18UFZ9fTbCpbNBSVLjuYTxXh9Z7LOEO7wNtNV6UMz6ctqGyO0y0TDLtpXhRR1BgJ+gye
3UQoZUnyIj0O2Y5BXm5URiu8tDsNIcWRd4mN2W5JRDNBQzOTI2o4azN8wSRGpyHkXZW9ZvqKzbBSe8xt
Sa+RFYYaFnfRdAGmbnuGWw0P3zEdV7Ua/rRbDWGJN8cCmV67kNNopWEvIwcnNThYDOb8uY3+Otvou9BL
xJE5K6wUS0I//G0JsDlxYZSRRxRcVSzEOeOh8YyHh0vitPuxN74qRWp/9uYKyq2TQLY4pMMh6zFIhTFz
oww+r5j00B6FL0yb6dj5hiNkKHqHzkHQCTZF68CniGKkCpefs4CmAQHNdHX/TINx0w720cT5TUo21oK8
PBmkG4NMTOFGz1fbYv0Rax6v3xfr2yvBaT9ocsbNz+vzxlrHoCO0BNKPQSocnm2DHWxUbGHp+xKa8k3a
uU6QFElVp98BUwK/b1mcwymArHHIZgyCzyAYsnhe7W5WIl5cQ3bP/MpmzzIXvYmW1ptSFPwhfUko/0EC
6XDIGodsxiAvVnwaLflA3GJbkViRBxqIW2sqBG3tQ0/hZkfqeNxX3+O++jLIBof0OGQYgxy5317GFViz
1Za2Ogv73fSipKmsEI1N0aFe2eQ4wLFR8NJb8CL7oVnO19mbtcvrUM2Pas0M/ihGZ8PYihlc3MGGBpJX
WeOQDQ7pdTbs+VRdztT2j4tKGjtRAquKP2uGjWsK7BQYM57ZmvDQ1rOisQVF41/f/Pjn1d++LKPZCqxu
nFEgjhRUywhHCxe0iSmoinFSFzbujNOmD4PMFOcL9XMcFwZFR7MscKQsNF8eFlIJsPUj3G6HAqsYQwB3
jtaEnFF78FAm90TjNpbRfu02ltE2xqMSBW07Q3421eTxH4GK7yPi2HyOxoK2Nekci3JNBYQ8VjDHs8nB
DtXJwguiBO+HEsP7ocdG/dwRv35HLKIrSQ5Ky35+iedEKJM/p4iKqnIKqLVZ8RG9WeaYz2d4C2e4qBsQ
WZxaP39n7wiuV86ogy9neBDCOXlYAmlTZaRXPtkroHBVw4tGKk0FyUgsI+cPNRI6SoO9HxOat5FylVfk
F+5j/gSsIgW2B3ZRl8jJDtpJGuioYXwgcE/f4JBDW/dcZete0Q6HrzeKa2kXMEEc4rRkNmjEHRG+khBA
+jHINydnFdVeYwMGwfWLjbMKpubOJvQmnx/TqaBqzdmiEbWZprWI+5e8x6654vr11I/11K5fsqO1tMWX
ZnPdXb/sXGfDOKnu2W9Q4N4WILJwA7Ztoj+ozIvwmBRh2RZKcMVOGgZXHCJsFVssbBWbYAMgDtnDVrH4
uCBVZpa45tiTmhsoZxtj0jiz/zsaOfliUugobYTvPQJIj0OGMQhuw9lotKXeia4ZbCm7/whXpixxt1JY
BASYCRm4yncqtjhrVxy1x9bIKE7SBmiQxdmhVBQBoocRwwji/Nl8tM9mRYu03lCe3nv6EOfL2iiGgloi
CCAdDtn3g0iwH8RktMG4qI28P11Rm1TV4BdVyoukyoIe1KWooFUmQrMaJZAOh6xxyGYMctTh6Ylqlf0r
DyIEbaxEy+OcgnzBuQJLc1zB/QThbYGDGbcSyDAGwZ8RCobDaS8ZhBSBhx+0LRqMF93vnEKOCEV03EbB
ouM2CgnliZMgkGF78CqoTom8j4qRDCcj/hReWaktOxdBsg5ZrzGmm85QX+OQDQ7pccjhZA/u9h5fpeID
46tvqqh9A5if7mmaZOa/QDL78hYvjf7BKbXIHkyK7idEaUmOlih2WtAaxKhgGU8uMOzY4Am2bIA7WuKS
jv2p3U62geJn9fKDu2Ol7h6EcBB8X4sRtlJI+C6uUveDUxDzaQtyhIO73BhhRlDfuWouvV+tIcCbYwGk
xyEDDtnikG4MclyZWlspvXoCNTGT4oimhMO14cltI1mTq9y2188luxw8drJ/ghzb2JqsbcvZWtgEL2xt
LWHiShhsMc8/dz08HgtE6HjMJ3gj46xHpWg+2Fi18Y7tUs1lu2/RWRrbUkL+TIPIAccpLL2ttRncYEsg
mzFIRf35nE+8AGVXOJ9zY157097tzrBCPAY5hmNCU0DrL6QIBmRYQScx7EOcQ3Oh4+SCbtxRngPQ3vJi
vUfUa3A5xNYiY/rpY5kMaSSLOpcJ146j0nFBfOm+cJ4TrJuPdNanN6BPl17WBAHRTmOs4XHXxw+QRYKY
lwyCUbyulUXPTIGToygCZnYmgTMlQh/WEkCHAtYoYHMIqGgFSjrhVkBi5lhSbGsGIckXd0HBbAzmwgsQ
HYxYjyDOPgkt+yTsnMggOlgFlZD1AeayCCBbHNLhkPUY5LxraHDXIAvUl4hFbFQYW6RC6NgigyYNMq2I
lojlrBY5CbXIVeh6kXivSgC6P1JITLDjhz8vGk5l0fAr9dORcXU532s8TnujlPN9TuBuIIFb9HF9bcr0
sKl4BVp3jBE1VQoZVaQJEP0BAp2nhhKq3O2REVhYbEigam+/7PBLprMXuCpq2NWFDAvQBJAehwxjEPxC
lUw41RuVsFiTadNcRKDLpPlDA3IEZ/XVCVNig0skTQFjbM5aZLqCrMZslAucbSeADDikG4Pgldqon5Lm
ZXp5RyXZfVqgqvGkIbh1qCDAE9eoXRyftkZV5s/FzR19goivKmbqPp0TZXrAYYke1QL4pOnxbk/A433h
zC1FaZYoKlojKTqECCdFw0HRIaJ9AHmf4O6T+ZwUfSJJ0TtBD2qz00jHZXwryfhWUgDZ4JB+DHLU5OcT
lf7nEzB2DteiUZNGgEfKMGMkhgAHPAf4ehcD6tJso6+LwINczJdmFGqOqUjma642VZWyVI+U9nhQuHCc
0gfIcQdVSxeUJkOV2uLydSTZpnubNTw1vMM9NdCVkvOxTurX8MxAcQO/6NBAtoEXXMaSgqSEiNDbmEuw
CQzB2aGCF9mOIPAP7YYL3Cvu+bmZzMNb10u0/44M0+wCL8ZFhxX4J4ii9j+fhPZ/Op5wYdm/LB12erFE
ToN0Hxln3cNOPwJIh0PWY5Dz1rTxeRVL3C1V7u4Vag8Lj2RthW7FwiEOEsh2DIJH3bTqbklfjbulyJHA
TiYwznfIjz6jdsXRJ3SdEAVLi/1c5+DRRzf6jM6Nk528PXo7/onR4ibPW0UNjG0uLl/CWwgKaw7yHHB3
bdSVkTy8+4tw+h9FpnNEVfPKhDsJc9ZVJUQcxH57uBEXQDY4pB+DVIw57OlyZ4UjDtsed/ZK4gfKXsGa
2TLcCEggGxzSj0EqLpvJ0KkSH4XXzWRatGi+Fsllc85zO4jFgtIEEC2JbY1eFgs1t6cpmBUFmy+7LxZM
DoPGdMVPOxqtD9hr6MEseJEehww4ZDsG+SazpqSr74XDpiTkd8n2O2aF7TcXWIHJKaFPh/OwA6/jFOBw
d6azyr0NlbuEDXrdXb88XLdVU5KDsTe8+5ZADsbegeGxN8FXXAuTBT5A8PMfuq7+/enIvG3WqOyS+ro5
nL9SbyYN+tiwU3g8Yizo4yGBbHFIh0P2aj3FPMmV9gpXgKWaHMXhjKz/X3ZOTjxd44kUOFY5oKOZDDOj
BYgeRmxHEMf2UVluC7S4jcpyHTwJ7rZJY+ooMF1f45ANDunHIHjb0abDhF77sbi/hGwSIwkwrLIu27+i
2YzG1rPN6MoyMpojx0wou4wtekdgO3l+f5XRxsI5TFvZxldh2oMtztcfkLE5o1EbEsi+bVu0cNSG4FW2
Y5AK4m6jTm9FUeGQfWsKG8Ghzyrh3TERLteChbKPkG9PC65nFbzg7c9LIlts1Og4SkI5fc4ltONwnAM8
E4fDQF0iNGzeOc+VOS/8TUR8EbfmVxMEbm+2RAWbBMewi5YAssYhGxzSj0Hw+6Y/7RWncPvvGws04une
g0zS8KMLROjoQgIZcEiHQ9ZjkG8lZF/z1riwdZ3E5ybOF0VY52F5G37gT0vd1/CLHJz3/tgeNy0e3mou
N4t2J9cSJ1EFZop1uCOTww1tnC8wbwt+6up6kxoGVmPuI8L+RMjAWjLPoxdI5IPx84fiZKJjB7JwJZA1
DtngkH4Mcp4Avv4EULKDF9R4NNbn+YHmyacIDpMTWfSxSAUdpSePTtJTSbrmFCdR4bJhysL+FFJXsmPp
MQ9dUiPsJBXiMqY5jfncCW93rjnVAk+7SUeTZ1cTGwuqGUWIHkYMMKIbQVR4ULP3J2tCLaxu9r45F+pr
kXzdaRyXXGBlebKo8R6FBGeaCN7Yvq4+VflQM7V7jGsaUTM11/lODzFYQ3smcK1a435SB0OMhO9L6qyx
zqZVzZus70/oR6bPXsW1ihjnnjLOPWWce8qwaxVH+Hl4fJWjbhxP1JjdN+9aFTqJvytbFa9q3P08Hzvs
vEUf1fZ4TMKUqskRmq8MS60Y4G5fZRprz9NYhdFCL2JFuDzfXMmThwPP0MWxALGGERsY0Y8gzgyKIzEo
1Jj3kjNVhXwfkyd0jxYogCuD4gvqZiN5lf3DPtpyXr2dyupN1DlEwxo2TyD9/gPg6Bsue95wKREVjrXf
6g+mrwUO6Euo7p9Cge2aPEwxJu/PEa5NpLHIBKoC21HSMIspBbbxKg4VgpItcNKlnXZQ2n+QLC5jIT6+
t2mLwwfpOlrmHrPktPf+qnv37sVhL1c1ygc2QtHBeQPRwcZLEbV5t+Rg36XoYN+lWGpmvfzVz3r5lWe9
UgqQoEcqVVrXfSOBRAz76dWlyeOudY0tj6Xzicbublcs6riLhnMpbNYfCXcuxQnwpapL8Ia/cgd0b7g1
B/RfSXRBTPM3FNYwO4aNVxxsvCKAbHFINwZZ4lQ+51IoJHUJFNOsIUrilGGXfzysWADpcciAQ7ZjkHMY
fXth9CLPDQHlIWr4CsTpgKsDi5ic0OcqVYSAZ7jXicWjz1Wi/G1OUVRzYpZjcYjs74LTCHtmj9YssUdt
YiSvMhxA6sbh9hQX47IhuG3OqPEnMhSno2Hqevm9zbiDHeYYXHE71F1usiGPo+cqh9jiejtqDqY5xCWH
JCIliSzDYv5a0qEDYEewGbljh+9SpvNH9y3BnM1VZl3u1L26hDoS16BXl6DKU8wKRnahoF21gzl4HANc
tBY2h3Y8aYhxrvO2gtAFmpIqt8YD6cb0OTvgkC0O6XDIGodsxiBnTUl7mhLRWkjwUVCyBkkF56ig5BEL
k7BKrhqXZ+NP8QopHJNn4xe2p3vzr9Wn93xYo2/dtHsBG41BecwoI0+A6GHEMIJQ9CLgE/AiGK2I49gQ
6KWMC9yhq1LG95d8meGFZcSTImKEoyKip+PbNi9ynC5v1ryk2+dOQk2lqNIbB4s6WUggHQ5Z45DNGOS4
Ite2GCR68laxFOtYne7PNAiKPjuF6AgOFj6gA5pnyGlSFehPzSLfn6pD/mQDK2gNSKHyXIKtKlxy8CQs
w1YVnrNTbQ3odGKOJ1vZhWtVZBAgsEt2GmEOvsan+2y6/dqm2yLmzvTa15une85qkwlnfUIXv6iFNn1c
dAHHHhnHhBrIOmsLaCDrgo1VS2afm5S+RsU+wOfcFhH/oQ8VOGbEqDDPCtaiAy0JpB+DVPgI+m+Cdbzs
oEoSoSfYi3HQyPm18DxVAunHIC+3ljNlbkv3lYsK3RbUfXbm6nJzd/Ni5dniNYSfDNNsBZAeh2xxSDcG
qbCw/OrXq9Q+fT1I6OtBgaRrC8ORRwkWPtmCe7YWhgfAlaMthGhzkhx0bi9WScAeyFFBouxCjDBhqwRU
cofT1bhM8zLXY69SMQzzDQcoqFLJPDfmfOWMZF+mYb3NDs8ddXjuKOxYMW3KOd6FYHlgi62/NL20hUFg
y8oueha0HrEomEdYVOojQAwwYgsjOhixHkEcWzLX1kJY13JoUX/OTiKYc16hVyGHtirkPe5RhzprRbhR
IZfPfnNN+M3J5sqCkKeYZnsqsrGW4EAlS3Ay1COkgvhYUruu8k6x8SgptmV5+BOJTIY0VhsU4RimiPIc
BIjtCOLbOy81c/GWoeTcCjjlxIZnc3Lcw4kLfeyKEBsY8XyZ/fCjQbX98TUUaej+TEM/irempMt1GmoJ
shZ1DZRANjikxyHbMchxT+7FbnIL97hHpvZKlJ4CP1mnYidbcDvZgtvJwmpSAeRgEZlI30+WTslPlr4a
P1nJIrI3xNMuLD4opIflgPbmAsTeU5JLRnfvghfZI9YRp6l+3ufZ4STLrSIVWU2ycJKFXS3KNK8uqehE
E+xxGwpsRR6KI5SKV+BBYiiTJlpuPDelxW2709xCprZo+V40XJlPyifHqO8s+VjgJXjA409swYO0UVM5
Z48+wzk1UzledJfz1zc//nn1t49v+P7wQdiSiECtQGD1Bc0K9nA6vADRw4hhBIGXdMtSQVF1f6GSXlF8
JcsNEthd+BLnLypLQheVZZofNey/BjpEL1TQ3M4SwIbs8ReoOs60JzDOlPoyLjvTfHlQz9NPQzKzCbRs
KIIFBO8zCXvYJIABBTx71JKpm+h/STaWv4qJvkwxtqAaV+TWWJcDtC8fDxmliDgLt/EuwObQzsGfIpy4
jmTbcBYG6YXLxsbck4JEcu69BsM2RFhzHuB8IpcIJuWGiA7xHyEVpNzYYAujyciNreneppNekooKiBku
bgFkg0N6HLIdg5xjW9qLbZFM4yUqY1IwwxMkaa5hxAZG9COIinPZpFOVSghH6mRSS9ait05gUsLe+NlL
o2Scx7xuRIjNHiKD1PPHnw28ND6+LUW3kXwKkRZa90Y10Zo0Q3a6vE3SCCHPAtXDegzy8jE57mFXuMVw
FNaU0FBpTQMZOsOTk4hgPM+2DSWTcyigcRLIZ5IA1ihgcwh4sbzL+GY9NmnLVBQ9cVNs7bDsBYPmYKou
aXtlFMFpggjRw4gBRmxhRDeCqKCaNLpziSe7cxE5lwg8S61VSIZwGeY5uZTg/CyXA7hVDBzQPAkXOVT6
nOYGS1zXBr0xg8rQ9aJwWYXmmCMc/h35bIHTtoSMBbMD8mb2zsILQiefHUPBZE/Y6MCjSZgSwIACts8B
KdRlZ36pf35y8jbsazo9aVi4eZZt7yRpxEmBs0R4HDYlRuUxFIhgf4WIGoiQz6kqf6rFdlgzgcq3tr+T
iCrrmuEDuzxYuyiBdDhkjUM2Y5CzqPIURZVXEsezqjSrA9qDzbAXmYVNVAWQzRgEX+Ilc7J+Z6LyTaY9
s7PHTce0vrF2OL0/wOOE5sx/gBxbSRgWa2W9qi+ISEu4qEe/4GoXTayyzzswRgJ952SQDQ7pcciAQ7Zj
kIpLHmTytIjgSjO2QmbttKR/nig2ICeV1AD0HiWBbHBIPwbB/Q+cSS12BHr2Bw8/YGMJl1eiZbU3VCUH
3+8HUNNeGaTHIVsc0o1BKpbWUKPS2tJaVuattSl3XqIUrCvxvStXyBG124gOzm/PMaAxrsEXdCwROaJ8
zsipfHNqwRbjjSTUY5EFmldg2xMl2IRSAOlxyDAGqRB8h1P0MJBSj1scW4gyEkvh+Yw6LxB7Dzhki0M6
HLJ+vhgsBZ2/fHiRby0iUTh/aZC5zNPsI2c8zT3DyVgwNkmE2MCIHkYMIwjFfOXXO/NJk1Ync/dQPPWF
vCMJt87P5tY9/iPgCf6IOBPYXp3AJhhxiLbSWUEzzeRhakNw6EiOKTOcJIMTOyYbgXOkRStWSL2gDWCT
FHK3MqNM3wwnjGY4YFSAGGDEdgRxbhyO2DgopmtMZxaYnBQmGxG1BIDDGSNqBxDr7LrabEtU7bqO0Jd8
9/GXfHG/uetuL6/f/vPyavgwRAnFU+QUvCsX3/3+3f8GAAD//1+jdaFWJQIA
`,
	},

	"/azure_instances.json": {
		name:    "azure_instances.json",
		local:   "_fixtures/azure_instances.json",
		size:    70021,
		modtime: 1559452593,
		compressed: `
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
`,
	},

	"/google_instances.json": {
		name:    "google_instances.json",
		local:   "_fixtures/google_instances.json",
		size:    17469,
		modtime: 1559452593,
		compressed: `
H4sIAAAAAAAC/+SbwY/aOBTG7/NXRDmXrN+z/WzPtSv1tFIP1V6qPWRMOkU7ARSCutNR//dVGAbCCBy/
kEWBnUMP4M8l5qdPnz+bl7skSZfVYrr29Sq9T77eJUmSvGz+TZLU53XxuKie0/sk/aMoF9VzsljWs3L2
s5imH95G1c/Lohkxh8n32eP3signjvZvL+a/F2U+n36uZr4ZR5l0GqzaDVgtF/Xbm1+3LybJS5L+XMw3
E69XkyJf1WqSpx+aj/s6FDI0DgwCidc/SH59CMofzpP7LvlW/dfuyfxyvfpcVH+W6X3iaPdyWZRvrxLu
1+GxNVrsXp3XPz4X1bfmkwAlnx5m9W+r9P27H1tfVPFPXeX7Ec1jbL7adw9yuCqtJd5//Lyuq9nDut7o
X3Zr0zxW8x+1vuQkSWfzVZ3PffHleVl8jOHmdSG2owhV+415Uf9YVH8ff7TtsF/7ZV5XVTGvPxXz9D75
lj+tirvWgKM0f1yUy3VdRODsl+sJqdM4Y6a11GiMJIkWUOtzwRaZBmGEMwbAgDWaxTVb7TvUQapJHaNa
m0w7R8JJo68Wb1IxeJ8G6YDv9opcFHSWbYc4VxmSVPrN7gScj7nVaCwoxeQ7VuZPyXoQrYBuHOQ4n1ZA
F8X3UzEvqvwpWa6r5WJVHKW3eZ5pXk0ncJpekQktlSPUA/IrGqts/wGTZP4EvnuCIN1wDG6ZcWwaY+B+
Wvy4BNoQQ/Yphg7AbtYgguzmuS7OdciWZaZQGtoTQOdzTVbZM6jmyn2XvIdfoxI37tdRVKMS47LrbzAp
Z75aBJ1a2AEgFkIafeCNkuvO/Bl8xAx8fxZ90vRt2rRgxeiLufUjTFZl/vQUxBqtO2ABh4DcKMfGOkLj
j2r46EJGzmGzkfy/k7tfiQuCy6s5VAhf0E47uWs5BmBXggMnYFs0CMHdB/In8N0TBBFXx9NzD3e2MYyX
xXS2Li+B+ZBVh2RZ9PYZL1512BDrWqKTKM7ZKr5DDQRpZ8h2EZr3kz2ckgVxtke7O7za4GyH6zk0jrfm
kHiaXcgMgG3tn+T5Ri0VqnZu4fo0W+879UGsJR5NIni9O0KJw8UQFOMlOxhAEKSzrd1UnwPE9wlCoT2I
5NzDEv4EefcE/AgCN5484si+cNzgZWugENskrbQKSdImiQ7QTQMaMsYJazdxXSA3QnD1vlMf3j0ePR4H
lSkhpBOE13uQCDRgum6vyHgPEkO0QyZIW1JiwJMYBAlOS2JCHivzp2R9kBbqxkmOS9ggxnzjI3SBSWZW
CnS0s+sB2mpDJEhru7VLJsdMte9Q97jHZOkGjNoNadTtFRnxkXkwllitXWvrNUQsMYAHHTg3lnD1vlPf
w8NJ3LiFRwVuEiM2cAwexRhnHFmyQOrAsc64CqKddlpobbWThp+3+RP47gmCaB9vSTLjnBWKWCc2alx7
Shwyb7cWZMR1dqgTxAwkmFbPMEAn2OxVnTBcymNl/pSsR/GHwt548RcXuFHY8eaQoF2DIOf2N/cGOThH
UOKcIMKfwHdPwPdrk924TUclEZONuvsLH9igIUNyF0bMAG2I1lIra9+iALcVYcp9l7yPadteQeQq7Tsy
iRwsyXirv7CTS4lSDnsFComUU3z/jpP5U7Ie2VretlVH9n7yslbNiiHBWyEKjTODtiHCajrr9wNcve/U
8y+MSHHbF0bifj4w4lP1UJetM5CaBvgtrkCDvUnmy32XvEeHLa+41nPD1Xpy1L1e0KE39zkRJFi5iZ4D
FB1iczTvtFKwqQrZFs3V+04936JNhkKA0xKu+HTGDpil2wsy3igdvg5FpIjOuw71nnUt0VnLvocdKfOn
ZPwLT0i3feEpstGj/yBK322/iHTlq3w5mz9+mZXF6/0q7RQQOkPg0rtfd/8GAAD//xw1KAo9RAAA
`,
	},

	"/": {
		name:  "/",
		local: `_fixtures`,
		isDir: true,
	},
}

var _escDirs = map[string][]os.FileInfo{

	"_fixtures": {
		_escData["/aws_instances.json"],
		_escData["/azure_instances.json"],
		_escData["/google_instances.json"],
	},
}

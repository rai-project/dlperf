// Code generated by "esc -o generated_data.go -pkg layer -prefix codegen -private codegen"; DO NOT EDIT.

package layer

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

	"/scope/base_prefix.tmpl": {
		name:    "base_prefix.tmpl",
		local:   "codegen/scope/base_prefix.tmpl",
		size:    458,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/5SQ30rDMBjF7/MUhw2kBekDzKtsKzp0Ufbnqgshdl9nHE21Sb3J+u4StKIwEe9yOCf8
Pn5jU+2pQi749C5XRYFsSrZ8qnV7FLomSMmsrsm96JJwrlaqKLKtNa8dfVWLuZQIjI33VBlLmOZidrPk
q9uzALUQD9uN4qvrdZJixwDeHlwSAnYs7nFCrY+kdHvoarLexaOAvk//T1CCL/NvmDhwSYifhoQTnhtj
MbrECFJGivPamxJvjdn/ReLzuZrdb8UmX61/U5M8DmkyWXvt6QKRQCkCw8czK5vOempdZqyj1ifhh4qh
/TSRXrGevQcAAP//NepZasoBAAA=
`,
	},

	"/scope/base_suffix.tmpl": {
		name:    "base_suffix.tmpl",
		local:   "codegen/scope/base_suffix.tmpl",
		size:    848,
		modtime: 1559014401,
		compressed: `
H4sIAAAAAAAC/8yST2vyQBCH7/spBvSwgfeNPVdaWOvSSk0QG0/rEtZmkixNtjF/DkL73ctqNbEK1ltv
YWZ+mecZthdhrA3CiPsPTx6bP4dCgDtC85rmqnzzVY4gJV05sCRCQF9lyXup6zSfoknqFG7vIEMDLtvX
K5AShAAdA65P529s+7CMrpz/90sCAMcA/ROCcOLPFkHI5o+hzzz+Qq8PdjKLCj1lGpUFOkfqDHdymFX4
TV8qk2CH3nr+cGz3BtybTVnA6eofHN0IpPwzfmDR0EQgJWm/SOsaqVrVmwK3qmNVq2BToDW9AEDPY9nq
/pfutvgBTVFgaZuhEH13YfS6wUNwMpbSGXbYeo2JML7wMjv61wZ2h/5dinySHppIxzAYAPfZaMrPz30F
AAD//4/GP2NQAwAA
`,
	},

	"/scope/batchnorm.tmpl": {
		name:    "batchnorm.tmpl",
		local:   "codegen/scope/batchnorm.tmpl",
		size:    367,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/3SQ3UoDMRCF7/MUc7FIBdkHWKvQ3S1YpCv05yqEME0Gu9j8mM4KRX13ibKC0l6eE3K+
M0dICQn9M0FhkZFPkaC6g7JFxs0p0hGUEkwuHpAJpmaw3tfIZt+F5JbBkmbYZe1DctoFS/fiyMi9gbfQ
W5ASirImb/YO00uHjkApnd0RV36bHzDESCk/aimLcuv714F+Py5apSa7UVXVmpHpCjKJruFdwAXQwsXD
9A+tyVeBUjf/a09+wm4FQD3vmoflbPWoz6fO2lY3T9tuM1+tL9cd8z7zxuRtXvIrAAD//4rYlMhvAQAA
`,
	},

	"/scope/conv_bias.tmpl": {
		name:    "conv_bias.tmpl",
		local:   "codegen/scope/conv_bias.tmpl",
		size:    644,
		modtime: 1559014909,
		compressed: `
H4sIAAAAAAAC/6yRX2uDMBTF3/spLp2M7sW+uz/QqtAyFsXoUwghq9ctTKON6aBs++7DOieFDlboW3Jz
7vndnMsYGKlfEJxcWmn3DYJ3D24grUz3DbbA+QSAMVAF4HZUuURWCNMUdVsbvza4kmUx7dVXqsixAD8L
CBE0i+MoSalIQ0KjREQx7Q1R5728tdKqDbzXKu/qjrtEvXmtpHk7MDgXXfUY/Am7pkHTPQrGHDfTarvD
38Z1wPnsebh5HrXS4vWBhDfwMQE4908/HSeGW1dNeXc0od8lB5w/zHrg7cDDssVL2g0JAixD4q+eFsmj
OO26CALhRxlJw4T+ndjo/3X+0lHnqoD5HP65+PH0HQAA///VR53ohAIAAA==
`,
	},

	"/scope/conv_bwd.tmpl": {
		name:    "conv_bwd.tmpl",
		local:   "codegen/scope/conv_bwd.tmpl",
		size:    858,
		modtime: 1559016003,
		compressed: `
H4sIAAAAAAAC/6ySX8vaMBTG7/0Uh3cy3sFLZbfOCf3jUIat9M9VCSFrTjUsTWqbKrLtu4+2aidTmLC7
NvnlPM95zklTqJjaIow5M8ycSoTpZ7A8Zlh8KrEGQkYAaQoiB9wPlOWzAuElRlXrytUVLpnMX3r6ncg5
5uAmnu/TKNlsgjCOaLzwoyCkwSbqC6LiPW6wKCUzCLNBxnK1OjhH3nqAj0AIZA1Xqj3VsjFCK+fIW5O2
3GpqunqyxgfgFyENVn+gnTRkA0SZ3OpKmF0xHwHUhhmRwUEL3uJjy0GV7QpWfe+6JoS2p7dR/ISmLLFq
L2majq1EiX2D14crj5DXb5e/6TQyzOD7Tgk/wI8RwLMpn1/cMbcqSjm7ceh2ORLydr/nt/OsriOiazte
zl97d58u5s4J/1/tv1UuewHgLHx3ubbDr/S+mO151A0SP16E0ePUh/q/nl9lVFzkMJnAP67z8PU7AAD/
/76hNZRaAwAA
`,
	},

	"/scope/conv_fwd.tmpl": {
		name:    "conv_fwd.tmpl",
		local:   "codegen/scope/conv_fwd.tmpl",
		size:    773,
		modtime: 1559015453,
		compressed: `
H4sIAAAAAAAC/6ySX2vCMBTF3/0UFyfDgdR354TaOpRhK/3zVELI2lsNS5Papg7Z9t1HrVpkDibsrb25
95xfTm4UQcHkGqGXMM30PkcYPYFhM82CfY4lENIBiCLgKeC27TIcliF0A5SlKixV4JyJtNt03/E0wRSs
0HYc6oerlesFPg1mju961F35jSDKpGnXmOWCaYRxXCVSWkrulKg0V/L5PTHFWlENcVukTKxVwfUmm3QA
Ss00j2GneFKL9owpyniTseLtAEgIrauX1J9Q5TkW9SGNop4RSr6t8Dy4sAnpv57+RiNfM433Byd8gI8O
wK2BHCeuwC2yXIwvCK06diBkcP3Og2Os5zTp0gzmk35D93iCQ1Hiv3v/dDk9IcB05ljzpem90Otmpm1T
yw2dYOb5v6fe6n/dvnUoE57CcAh/3Lz26zsAAP//K6U0AAUDAAA=
`,
	},

	"/scope/dropout.tmpl": {
		name:    "dropout.tmpl",
		local:   "codegen/scope/dropout.tmpl",
		size:    709,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/6yQTUvzQBSF9/MrLrS8JIs33adayBcaNKPkYzUOw9jcarAZYzIRiva/y1gSLLTSgsvc
8Jxz5iGMQSvVE8K0lFrqTYPgXoITSi3zTYMdcE46LXW1hPfXqgTGYOr4qJbPtWxfqKwROBfmOvDO9/ET
+qbB1vwUjE2dQlVvPY5gHHJuPQ5frptpqfEfmCa04YPAkaK4btYXe22BmQmcL6wdPCcAfkSD68RLb8Th
FC8MRXBX0DxKs+PzhrytkYSqNCrIScLGfusPddn/Fw8nvC2m90UuvPRKUC+JMutcbCSKDhOpernOqxot
e/7DwqRXJa72M3+LPBfYTSdbMkFVViuYzSCinn8bHeTIVwAAAP//UFK1K8UCAAA=
`,
	},

	"/scope/pooling.tmpl": {
		name:    "pooling.tmpl",
		local:   "codegen/scope/pooling.tmpl",
		size:    375,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/3SQ3UrDQBCF7/MU5yJIBckDxCq0ScEijdKfq7AsY3aowWR3TSdCUd9dNqFKwV7OLPt9
Z04ElCU6sntGbEhIjp6R3iHJSWh79HyAUhEg3PqGhDGtemPts3NNbfcrZ1gL/Djp1hm+j4CDkNQVPlxt
Aj5O5myr15a6t4JahlI6bE+6ZFh+ofeeu/CoyzJOdrZ+7/n34zJXavJymtJ0IyR8NZj4Gp8RgAuqZeub
6ZkvC3dBqZvz4JMRdjuw5osie1jN1o/6f+osz3X2tCu2i/XmcuA/4vfYNFsT+vwJAAD//5fpR2F3AQAA
`,
	},

	"/scope/relu.tmpl": {
		name:    "relu.tmpl",
		local:   "codegen/scope/relu.tmpl",
		size:    370,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/3SQX0vDQBDE3/Mp5iFIBckHiFVIk4JFGqF/no7jWHOLBnuXM90Uivrd5ZQUxPZxZ9n5
zU6iFHryL4zUkpAcAyO/Q1aR0OYYeA+tE2EXdiSMaTNY74tG2gNJ2/llZ9kI6CQY11m+T/ZC0jY4dK2F
UkizGfvm1VH/VpNjaG2iOgKzH/ETQwjcx6VRKs22vn0f+HS4qLSePI9Tnq+FhK8QSXyNjwQXQAsXdtM/
tDL+Ba1v/uWe/LrdJsBsXpcPy2L1aM7bFlVlyqdtvZmv1pfzjn5fsWb2Npb5HQAA//9dLGtNcgEAAA==
`,
	},

	"/scope/softmax.tmpl": {
		name:    "softmax.tmpl",
		local:   "codegen/scope/softmax.tmpl",
		size:    1540,
		modtime: 1558740137,
		compressed: `
H4sIAAAAAAAC/8RUYW/aMBD97l9xUtGUSIz2M3SVTOJ2aCRUxJEmBcsy+GijEScDZ1q19b9PgYWFjopU
2kS+5Xz33r17Z5MkgbUyDwgdrayyTwVC/wP0fGUVfypwA0IQi1mxUhbhelFqY6J8aTP1na4e8nVqHzNp
YbMLSVXHutBMDXKNjaws13hDNlbZdAHf8lRDkkCnN0SzeMzU+kuoMgQhZBWtu+ptgz+hLApcV4cySTq9
2KRfS9wXjnwhnHn91+9HVll8BxUTuvCDwCtEo6xYXR+weZV4EKJ7TNqBDmeHPiAAQxZ6HwM6/SSP01Df
l94kDjmbRq/3X+M9V96g0ZUDFxqXqcFDhr8Irpx5F6LJLQ/oZxlMfObCrGHwXsHW4b19lcUNYM6C+zHl
7CVWdzu7PxhCuDMC8P7mlOpReB9zSad3MqQBi5y3lu0r4g0GypRqxdMMHXewE/d7Qi1H5Mxd+J/f7OUa
HDfJi/0wlPV4b2nE3cHZ6KnnxVPK2RlbGE/uXNLmLTqxNc4/fEjcQfMClkbj8oSwxs6+tWB3N9pVXbVL
I8/kAo1Ol3B5CSykwzE7nvcrAAD//+QaFkQEBgAA
`,
	},

	"/tensorflow/base_prefix.py": {
		name:    "base_prefix.py",
		local:   "codegen/tensorflow/base_prefix.py",
		size:    408,
		modtime: 1558738826,
		compressed: `
H4sIAAAAAAAC/3TPzUrEMBDA8XufYoiXXZHiWfCgbZGC1IW6vYiE0E3WQDMT8rHSt5fWthsP3ob5/1oy
2lhyATAaO4LwgDZbVuTXyY/bGCR6cmqg7wkHlSlHBnoyhjC3wgnjYaG3f1oMekhSRj6XeNGO8IMVx/KJ
d3VbP79WvKy6uqha9gmPwO5Zlt2AdRrDjr21D8DuptfkdhBBkTP7az2M4YtwExfpvCZMQDOdOHe0OecL
4Dwh79t1swvqH/dyOM7gLAM/28hRGLnbX8G07+NJrB+njRWxbBrofsv6lz6eEFP+EwAA//+73BsbmAEA
AA==
`,
	},

	"/": {
		name:  "/",
		local: `codegen`,
		isDir: true,
	},

	"/scope": {
		name:  "scope",
		local: `codegen/scope`,
		isDir: true,
	},

	"/tensorflow": {
		name:  "tensorflow",
		local: `codegen/tensorflow`,
		isDir: true,
	},
}

var _escDirs = map[string][]os.FileInfo{

	"codegen": {
		_escData["/scope"],
		_escData["/tensorflow"],
	},

	"codegen/scope": {
		_escData["/scope/base_prefix.tmpl"],
		_escData["/scope/base_suffix.tmpl"],
		_escData["/scope/batchnorm.tmpl"],
		_escData["/scope/conv_bias.tmpl"],
		_escData["/scope/conv_bwd.tmpl"],
		_escData["/scope/conv_fwd.tmpl"],
		_escData["/scope/dropout.tmpl"],
		_escData["/scope/pooling.tmpl"],
		_escData["/scope/relu.tmpl"],
		_escData["/scope/softmax.tmpl"],
	},

	"codegen/tensorflow": {
		_escData["/tensorflow/base_prefix.py"],
	},
}

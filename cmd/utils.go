package cmd

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/acarl005/stripansi"
	"github.com/alecthomas/repr"
	"github.com/cheggaaa/pb"
	"github.com/k0kubun/pp"
	zglob "github.com/mattn/go-zglob"
	homedir "github.com/mitchellh/go-homedir"
	"github.com/pkg/errors"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/utils"
	"golang.org/x/sync/errgroup"
)

// location of dot executable for converting from .dot to .svg
// it's usually at: /usr/bin/dot
var (
	dotExe         string
	dotOutputCount int = 0
)

func dotToImage(dot []byte) (string, error) {
	if dotExe == "" {
		dot, err := exec.LookPath("dot")
		if err != nil {
			log.Fatalln("unable to find program 'dot', please install it or check your PATH")
		}
		dotExe = dot
	}
	dotOutputCount++

	img := filepath.Join(os.TempDir(), fmt.Sprintf("dlperf_%d.png", dotOutputCount))
	// img := filepath.Join("/tmp", fmt.Sprintf("dlperf.png"))
	cmd := exec.Command(dotExe, "-Tpng", "-o", img)
	cmd.Stdin = bytes.NewReader(dot)
	if err := cmd.Run(); err != nil {
		bts, e := cmd.CombinedOutput()
		if e == nil {
			return "", errors.Wrap(err, string(bts))
		}
		return "", err
	}
	return img, nil
}

func newProgress(prefix string, count int) *pb.ProgressBar {
	bar := pb.New(count).Prefix(prefix)
	bar.SetRefreshRate(time.Millisecond * 100)
	bar.Start()
	return bar
}

func expandModelPath(modelPath string) string {
	if dir, err := homedir.Expand(modelPath); err == nil {
		modelPath = dir
	}

	if modelPath == "" {
		modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
	} else {
		s, err := filepath.Abs(modelPath)
		if err == nil {
			modelPath = s
		}
	}
	return modelPath
}

func getModelsIn(modelPath string) ([]string, error) {
	if !com.IsFile(modelPath) && !com.IsDir(modelPath) {
		return nil, errors.Errorf("file %v does not exist", modelPath)
	}

	if com.IsFile(modelPath) {
		return []string{modelPath}, nil
	}

	modelPaths0, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
	if err != nil {
		return nil, err
	}
	modelPaths := []string{}
	for _, modelPath := range modelPaths0 {
		modelName := getModelName(modelPath)
		if strings.HasPrefix(modelName, ".") || strings.HasPrefix(filepath.Base(modelPath), ".") {
			continue
		}
		modelPaths = append(modelPaths, modelPath)
	}
	return modelPaths, nil
}

func readModels(modelPath string) ([]*onnx.Onnx, error) {
	if !com.IsFile(modelPath) && !com.IsDir(modelPath) {
		return nil, errors.Errorf("file %v does not exist", modelPath)
	}

	dlperf.SetBatchSize(batchSize)

	if com.IsFile(modelPath) {

		model, err := onnx.New(modelPath, batchSize)
		if err != nil {
			return nil, err
		}
		return []*onnx.Onnx{model}, nil
	}

	// is a directory

	modelPaths0, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
	if err != nil {
		return nil, err
	}
	modelPaths := []string{}
	for _, modelPath := range modelPaths0 {
		modelName := getModelName(modelPath)
		if strings.HasPrefix(modelName, ".") || strings.HasPrefix(filepath.Base(modelPath), ".") {
			continue
		}
		modelPaths = append(modelPaths, modelPath)
	}

	models := []*onnx.Onnx{}

	modelReadProgress := newProgress("> Reading models", len(modelPaths))
	var mut sync.Mutex
	g, _ := errgroup.WithContext(context.Background())
	for ii := range modelPaths {
		modelPath := modelPaths[ii]

		g.Go(func() error {

			path := modelPath

			defer modelReadProgress.Increment()
			defer func() {
				if r := recover(); r != nil {
					pp.Println("[PANIC] while processing " + path + " [error = " + stripansi.Strip(repr.String(r)) + "]")
				}
			}()
			// pp.Println(path)
			model, err := onnx.New(path, batchSize)
			if err != nil {
				log.Fatal(err)
			}
			mut.Lock()
			defer mut.Unlock()
			models = append(models, model)

			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	modelReadProgress.Finish()
	return models, nil

}

func getSrcPath(importPath string) (appPath string) {
	paths := com.GetGOPATHs()
	for _, p := range paths {
		d := filepath.Join(p, "src", importPath)
		if com.IsExist(d) {
			appPath = d
			break
		}
	}

	if len(appPath) == 0 {
		appPath = filepath.Join(goPath, "src", importPath)
	}

	return appPath
}

func flopsToString(e int64, humanFlops bool) string {
	if humanFlops {
		return utils.Flops(uint64(e))
	}
	return fmt.Sprintf("%v", e)
}

func getModelName(modelPath string) string {
	return iGetModelName(modelPath, "")
}

func findModelNameFile(dir string, level int) (string, error) {
	if level < 0 {
		return "", errors.New("unable to find model_name file")
	}
	pth := filepath.Join(dir, "model_name")
	if com.IsFile(pth) {
		bts, err := ioutil.ReadFile(pth)
		if err != nil {
			return "", err
		}
		return string(bts), err
	}
	return findModelNameFile(filepath.Dir(dir), level-1)
}

func iGetModelName(modelPath, suffix string) string {
	name, err := findModelNameFile(filepath.Dir(modelPath), 4 /* levels */)
	if err == nil {
		return removeSpace(name)
	}
	name = strings.TrimSuffix(filepath.Base(modelPath), ".onnx")
	if name != "model" && name != "model_inferred" {
		return removeSpace(name + suffix)
	}
	if suffix == "" && name == "model_inferred" {
		suffix = "_inferred"
	}
	return iGetModelName(path.Dir(modelPath), suffix)
}

func removeSpace(s string) string {
	s = strings.TrimSuffix(s, "\n")
	s = strings.TrimSpace(s)
	return s
}

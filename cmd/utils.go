package cmd

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Unknwon/com"
	"github.com/cheggaaa/pb"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
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
		return "", err
	}
	return img, nil
}

func newProgress(prefix string, count int) *pb.ProgressBar {
	// get the new original progress bar.
	//bar := pb.New(count).Prefix(prefix)
	// TODO: set prefix of bar
	bar := pb.New(count)
	//bar.Set("prefix", prefix)

	// Refresh rate for progress bar is set to 100 milliseconds.
	bar.SetRefreshRate(time.Millisecond * 100)

	// Use different unicodes for Linux, OS X and Windows.
	switch runtime.GOOS {
	case "linux":
		// Need to add '\x00' as delimiter for unicode characters.
		bar.Format("┃\x00▓\x00█\x00░\x00┃")
	case "darwin":
		// Need to add '\x00' as delimiter for unicode characters.
		bar.Format(" \x00▓\x00 \x00░\x00 ")
	default:
		// Default to non unicode characters.
		bar.Format("[=> ]")
	}
	bar.Start()
	return bar
}

func readModels(modelPath string) ([]*onnx.Onnx, error) {

	if !com.IsFile(modelPath) && !com.IsDir(modelPath) {
		return nil, errors.Errorf("file %v does not exist", modelPath)
	}

	if com.IsFile(modelPath) {
		model, err := onnx.New(modelPath)
		if err != nil {
			return nil, err
		}
		return []*onnx.Onnx{model}, nil
	}

	// is a directory

	modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
	if err != nil {
		return nil, err
	}
	models := make([]*onnx.Onnx, len(modelPaths))

	modelReadProgress := newProgress("reading models", len(modelPaths))

	g, _ := errgroup.WithContext(context.Background())
	for ii := range modelPaths {
		idx := ii
		g.Go(func() error {
			defer modelReadProgress.Increment()
			path := modelPaths[idx]
			// pp.Println(path)
			model, err := onnx.New(path)
			if err != nil {
				return err
			}
			models[idx] = model
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

func iGetModelName(modelPath, suffix string) string {
	name := strings.TrimSuffix(filepath.Base(modelPath), ".onnx")
	if name != "model" && name != "model_inferred" {
		return name + suffix
	}
	if suffix == "" && name == "model_inferred" {
		suffix = "_inferred"
	}
	return iGetModelName(path.Dir(modelPath), suffix)
}

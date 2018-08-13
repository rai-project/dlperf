package cmd

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/onnx"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

// location of dot executable for converting from .dot to .svg
// it's usually at: /usr/bin/dot
var dotExe string

func dotToImage(dot []byte) (string, error) {
	if dotExe == "" {
		dot, err := exec.LookPath("dot")
		if err != nil {
			log.Fatalln("unable to find program 'dot', please install it or check your PATH")
		}
		dotExe = dot
	}

	img := filepath.Join(os.TempDir(), fmt.Sprintf("dlperf.png"))
	cmd := exec.Command(dotExe, "-Tpng", "-o", img)
	cmd.Stdin = bytes.NewReader(dot)
	if err := cmd.Run(); err != nil {
		return "", err
	}
	return img, nil
}

// todotCmd represents the todot command
var todotCmd = &cobra.Command{
	Use:     "todot",
	Aliases: []string{"dot"},
	Short:   "Converts graph to dot format",
	RunE: func(cmd *cobra.Command, args []string) error {

		if modelPath == "" {
			modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
		} else {
			s, err := filepath.Abs(modelPath)
			if err == nil {
				modelPath = s
			}
		}

		if !com.IsFile(modelPath) {
			return errors.Errorf("file %v does not exist", modelPath)
		}

		model, err := onnx.New(modelPath)
		if err != nil {
			return err
		}

		grph := model.ToGraph()

		dotEnc, err := dot.Marshal(grph, model.GetName(), "", "  ", true)
		if err != nil {
			return err
		}

		// println(string(dotEnc))

		// dominators := model.Dominators()

		// pp.Println(dominators)

		subgrphs, err := model.FindGraphGroups()
		dotEnc, err = dot.Marshal(subgrphs[1], model.GetName(), "", "  ", true)

		img, err := dotToImage(dotEnc)
		if err != nil {
			return err
		}

		println(img)

		return nil
	},
}

func init() {
	rootCmd.AddCommand(todotCmd)
}

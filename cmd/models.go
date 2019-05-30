package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var modelURLsCmd = &cobra.Command{
	Use:     "modelurls",
	Aliases: []string{"models"},
	Short:   "Prints the model urls",
	RunE: func(cmd *cobra.Command, args []string) error {
		for _, model := range modelURLs {
			fmt.Printf("(\"%s\", \"%s\"), \n", model.Name, model.URL)
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(modelURLsCmd)
}

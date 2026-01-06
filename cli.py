import click
import asyncio
from gitbook_downloader import GitbookDownloader


@click.group()
def cli():
    """GitBook Downloader CLI"""
    pass


@cli.command()
@click.argument("url")
@click.option("--output", "-o", default=None, help="Output markdown file")
@click.option("--native", "-n", is_flag=True, help="Request native markdown")
@click.option("--site-prefix", "-s", default=None, help="Only download URLs starting with this prefix")
@click.option("--image", "-i", "download_images", is_flag=True, help="Download images and rewrite references")
@click.option(
    "--image-ignore-prefix",
    "-I",
    is_flag=True,
    help="When downloading images, ignore site-prefix restriction",
)
@click.option(
    "--image-dir",
    "-d",
    default="images",
    show_default=True,
    help="Directory to store downloaded images",
)
def download(url, output, native, site_prefix, download_images, image_ignore_prefix, image_dir):
    """Download a GitBook by URL and save as markdown."""

    async def run():
        downloader = GitbookDownloader(
            url,
            native_md=native,
            site_prefix=site_prefix,
            download_images=download_images,
            ignore_image_prefix=image_ignore_prefix,
            image_output_dir=image_dir,
        )
        markdown = await downloader.download()
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(markdown)
            click.echo(f"Saved to {output}")
        else:
            click.echo(markdown)

    asyncio.run(run())


if __name__ == "__main__":
    cli()

document.addEventListener("DOMContentLoaded", function () {
    const seeMoreLink = document.querySelector(".see-more-link");
    const hiddenPosts = document.querySelectorAll(".post.hidden");

    seeMoreLink.addEventListener("click", function (e) {
        e.preventDefault();

        hiddenPosts.forEach(post => {
            post.classList.remove("hidden");
        });

        seeMoreLink.style.display = "none";
    });
});
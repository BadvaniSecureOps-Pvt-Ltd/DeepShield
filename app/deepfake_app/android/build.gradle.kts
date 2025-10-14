buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.6.0") // Android Gradle Plugin
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.1.0") // Kotlin plugin
        classpath("com.google.gms:google-services:4.4.2") // Firebase plugin
    }
}

// ✅ Redirect build directories outside the project folder
val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

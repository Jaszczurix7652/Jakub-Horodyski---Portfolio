import { ChakraProvider, extendTheme, Heading, Container, Input, Button, Wrap, Image, Center, Box, Text, useToast, VStack } from "@chakra-ui/react";
import axios from "axios";
import { useState } from "react";

// Custom theme to modify global styles and colors
const theme = extendTheme({
  styles: {
    global: {
      html: {
        height: "100%",
      },
      '#root': {
        height: "100%",
      },
      body: {
        minH: "100vh",
        bgGradient: "linear(to-br, #8F16C8, #24CB61)",
        color: "white",
        lineHeight: "base",
      },
    },
  },
});

const App = () => {
  const [prompt, updatePrompt] = useState('');
  const [loading, updateLoading] = useState(false);
  const [stableDiffusionImage, setStableDiffusionImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const toast = useToast();
  const [textToImage, setTextToImage] = useState(null);

  const generateStableDiffusion = async (prompt) => {
    if (!prompt) {
      toast({
        title: "Błąd",
        description: "Należy wpisać opis obrazu przed wyborem generatora",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    updateLoading(true);
    try {
      const result = await axios.get(`http://127.0.0.1:8000/?prompt=${prompt}`);
      setStableDiffusionImage(result.data);
      setImageUrl(null);
    } catch (error) {
      toast({
        title: "Failed to generate image",
        description: "Stable Diffusion API error",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
    updateLoading(false);
  };

  const generateDALLE = async (prompt) => {
    if (!prompt) {
      toast({
        title: "Błąd",
        description: "Należy wpisać opis obrazu przed wyborem generatora",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    updateLoading(true);
    try {
      const result = await axios.post("http://127.0.0.1:8000/dalle", { prompt: prompt });
      setImageUrl(result.data.image_url);
      setStableDiffusionImage(null); 
    } catch (error) {
      toast({
        title: "Failed to generate image",
        description: "DALL-E API error",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
    updateLoading(false);
  };

  const generateTextToImage = async (prompt) => {
    if (!prompt) {
      toast({
        title: "Error",
        description: "Należy wpisać opis obrazu przed wyborem generatora",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    updateLoading(true);
    try {
      const result = await axios.post("http://127.0.0.1:8000/text-to-image", { prompt: prompt });
      setTextToImage(result.data.image_url);
      setStableDiffusionImage(null); 
      setImageUrl(null);
    } catch (error) {
      toast({
        title: "Failed to generate image",
        description: "Text-to-Image API error",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
    updateLoading(false);
  };


  return (
    <ChakraProvider theme={theme}>
      <Box padding={5}>
        <Container maxW="container.md" boxShadow="xl" rounded="md" p={5} mt={10}
          bg="rgba(45, 55, 72, 0.7)" 
          backdropFilter="blur(10px)" 
          color="white" 
        >
          <VStack spacing={6}>
            <Heading as="h1" size="xl" mb={6} textAlign="center">Generatory obrazów</Heading>
            <Text fontSize="lg" mb={4}>Przed generowaniem obrazu wpisz jego opis:</Text>
            <Wrap spacing={4} justify="center" mb={4}>
              <Input
                placeholder="Wpisz tutaj swój opis..."
                value={prompt}
                onChange={(e) => updatePrompt(e.target.value)}
                size="md"
                focusBorderColor="blue.300"
                _placeholder={{ color: 'gray.500' }}
              />
              <Button
                onClick={() => generateStableDiffusion(prompt)}
                colorScheme="yellow"
                isLoading={loading && !imageUrl && !textToImage}
              >
                Stable Diffusion v1.5
              </Button>
              <Button
                onClick={() => generateDALLE(prompt)}
                colorScheme="blue"
                isLoading={loading && !stableDiffusionImage && !textToImage}
              >
                DALL-E 2
              </Button>
              <Button
                onClick={() => generateTextToImage(prompt)}
                colorScheme="green"
                isLoading={loading && !textToImage}
              >
                Autorski model
              </Button>
            </Wrap>
            {loading && <Text>Ładowanie...</Text>}
            {!loading && stableDiffusionImage && (
              <Center mt={4}>
                <Image src={`data:image/png;base64,${stableDiffusionImage}`} alt="Generated Image" boxShadow="lg" maxW="full" />
              </Center>
            )}
            {!loading && imageUrl && (
              <Center mt={4}>
                <Image src={imageUrl} alt="Generated Image" boxShadow="lg" maxW="full" />
              </Center>
            )}
            {!loading && textToImage && (
              <Center mt={4}>
                <Image src={textToImage} alt="Generated Image" boxShadow="lg" maxW="full" />
              </Center>
            )}
          </VStack>
        </Container>
      </Box>
    </ChakraProvider>
  );
};  


export default App;

